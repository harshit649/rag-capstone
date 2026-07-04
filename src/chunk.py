from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from generate import generate

print(">>> SCRIPT STARTED")   # put this as the FIRST line after imports
model = SentenceTransformer("all-MiniLM-L6-v2")

def dot(a, b):
    return np.sum(np.array(a) * np.array(b))

def cosine(a, b):
    return dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, docs, k=3):
    q = model.encode(query)
    d_vecs = [model.encode(d) for d in docs]      # embed each chunk
    scores = [cosine(q, d) for d in d_vecs]        # query vs each chunk
    return sorted(zip(scores, docs), reverse=True)[:k]

def read_pdf(path):
    reader = PdfReader(path)              # open + parse the PDF into pages
    text = ""
    for page in reader.pages:             # reader.pages = list of page objects
        text += page.extract_text() or "" # pull text from each page, append it
    return text

def chunk_text(text, size, overlap):
    words = text.split()
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i+ size]))
        if i + size>= len(words):
            break
    
    return chunks

import re
def clean_text(text):
    pos = text.find("Abstract")
    if pos != -1:
        text = text[pos + len("Abstract"):]

    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub("[∗†‡]", "", text)
    return text


paper = read_pdf("data/papers/attention.pdf")

cleaned_words = clean_text(paper)

chunks = chunk_text(cleaned_words, size=120, overlap=20)


def answer(query):
    results = retrieve(query, chunks, k=3)
    context = "\n\n".join(text for score, text in results)
    prompt = f"""Use ONLY the context below to answer. If the answer isn't there, say you don't know
    Context
    {context}
    Question
    {query}"""
    return generate(prompt)


print(answer("why do we scale the dot products by the square root of dk?"))
print(answer("what optimizer did they use?"))


eval_set = [
    ("why do we scale the dot products by √dk?",  15),
    ("what optimizer did they use?",              31),
    ("how many layers are in the encoder?",       10),
    ("how many attention heads?",                 17),
    ("what dataset was used for English-German?", 30),
]

# def find(keyword):
#     for i, c in enumerate(chunks):
#         if keyword.lower() in c.lower():
#             print(i, "→", c[:160])
#             print("---")

# find("optimiser Adam") #             # optimizer question
# # find("N = 6 ")  #done         # encoder layers  (try "identical layers" too)
# find("parallel attention")           # attention heads (try "parallel attention" too)
# find("4.5 million")   # dataset (try just "4.5 million" too)
# # find("1√dk")    # done        # the scaling one (you saw this exact text earlier)
# find("attention heads")
# find("English german")


# evaluation metrics
def recall_at_k(eval_set, k=3):
    hits = 0
    for question, correct_id in eval_set:
        results = retrieve(question, chunks, k)
        retrieve_text = [text for score, text in results]
        if chunks[correct_id] in retrieve_text:
            hits +=1
        else:
            print(f"question miss {question} wanted chunk correct id {correct_id}")
    
    recall = hits / len(eval_set)
    print(f"recall@{k} = {recall}")

# recall_at_k(eval_set, k=3)

# mrr
def mrr(eval_set):
    total_rr = 0
    for question, correct_id in eval_set:
        results = retrieve(question, chunks, k=len(chunks))   # ALL chunks, fully ranked
        retrieved_texts = [text for score, text in results]  # retrive text is actaully sorting the chunks and tehn giving
        rank = retrieved_texts.index(chunks[correct_id]) + 1
        rr = 1/ rank
        total_rr += rr
        print(f"rank {rank:2d}  ←  {question}")   # ← see where each lands
    mrr = total_rr / len(eval_set)
    print(f"MRR = {mrr}")

# mrr(eval_set)


def faithfulness(query):
    results = retrieve(query, chunks, k=3)
    context = "\n\n".join(text for score, text in results)
    answer_text = answer(query)                 # the answer being judged
    judge_prompt = f"""You are evaluating whether an answer is faithful to a context.
    An answer is faithful ONLY if every claim it makes is supported by the context.

    Context:
    {context}

    Answer:
    {answer_text}

    Is every claim in the answer supported by the context above?
    Reply with exactly one word: FAITHFUL or UNFAITHFUL."""
    verdict = generate(judge_prompt)            # LLM as judge
    print(f"{verdict.strip()}  ←  {query}")
    return verdict

# faithfulness("what optimizer did they use?")
# faithfulness("why do we scale the dot products by √dk?")
# faithfulness("how many layers are in the encoder?")
# faithfulness("how many attention heads?")


def judge(context, answer_text):
    judge_prompt = f"""You are evaluating whether an answer is faithful to a context.
An answer is faithful ONLY if every claim it makes is supported by the context.

Context:
{context}

Answer:
{answer_text}

Is every claim in the answer supported by the context above?
Reply with exactly one word: FAITHFUL or UNFAITHFUL."""
    return generate(judge_prompt).strip()

# real context for the optimizer question
ctx = "\n\n".join(text for score, text in retrieve("what optimizer did they use?", chunks, k=3))

# 1) a faithful answer (should say FAITHFUL)
# print(judge(ctx, "They used the Adam optimizer with β1=0.9, β2=0.98."))

# # 2) a HALLUCINATED answer — adds a claim the context never makes (should say UNFAITHFUL)
# print(judge(ctx, "They used the Adam optimizer, which Google invented in 2017 specifically for training Transformers."))



## ragas

# building eveal data set
eval_data = []
for question, correct_id in eval_set:
    response = answer(question)
    contexts = [texts for score, texts in retrieve(question , chunks , k=3)]
    eval_data.append({"question": question, "response": response, "contexts": contexts})


print("-----------------------")
print("eval data 0", eval_data[0])


# ---- RAGAS ----
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import llm_factory
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",   # Groq's OpenAI-compatible endpoint
)
evaluator_llm = llm_factory("llama-3.3-70b-versatile", client=groq_client)

samples = [
    SingleTurnSample(
        user_input=row["question"],
        response=row["response"],
        retrieved_contexts=row["contexts"],
    )
    for row in eval_data
]
dataset = EvaluationDataset(samples=samples)

result = evaluate(dataset=dataset, metrics=[Faithfulness(llm=evaluator_llm)])
print(result)
