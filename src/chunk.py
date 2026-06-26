from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from generate import generate

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
    # words = text.split()
    # position = words.index("Abstract")
    # print("position", position)
    # cleaned_words = " ".join(words[position+1:])
    # return cleaned_words
    pos = text.find("Abstract")
    if pos != -1:
        text = text[pos + len("Abstract"):]

    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub("[∗†‡]", "", text)
    return text

# text = "the cat sat on the mat near the door"
# print(chunk_text(text, 4, 2))

# text = ("Retrieval augmented generation grounds a language model by fetching relevant "
#         "documents before it answers which reduces hallucination and lets the model use "
#         "knowledge that was never part of its original training data at all")

# print(chunk_text(text, size=10, overlap=3))


paper = read_pdf("data/papers/attention.pdf")
# print("characters:", len(paper))
# print(paper[:800])

cleaned_words = clean_text(paper)
# print("cleaned words", cleaned_words)

chunks = chunk_text(cleaned_words, size=120, overlap=20)
# print("num chunks:", len(chunks))
# print("---")
# print(chunks[0])
# print(chunks[20])


results = retrieve("why do we scale the dot products by the square root of dk?", chunks, k=3)
for score, chunk in results:
    print(round(float(score), 3), "→", chunk[:200])
    print("---")


context = "\n\n".join(text for score, text in results)
question = "{what is dot product}"
prompt = f"""Use ONLY the context below to answer. If the answer isn't there, say you don't know
Context
{context}
Question
{question}"""
print(generate(prompt))
            