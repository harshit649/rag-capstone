from ingest import chunks
from retriever import retrieve, hybrid_retrieve, retrieve_chroma, rerank_retrieve
from generator import answer

eval_set = [
    ("why do we scale the dot products by √dk?",  15),
    ("what optimizer did they use?",              31),
    ("how many layers are in the encoder?",       10),
    ("how many attention heads?",                 17),
    ("what dataset was used for English-German?", 29),
]


def recall_at_k(eval_set, k=3):
    hits = 0
    for question, correct_id in eval_set:
        results = retrieve(question, chunks, k)
        retrieved = [text for score, text in results]
        if chunks[correct_id] in retrieved:
            hits += 1
        else:
            print(f"miss: {question}")
    print(f"recall@{k} = {hits/len(eval_set)}")


def mrr(eval_set):
    total_rr = 0
    for question, correct_id in eval_set:
        results = retrieve(question, chunks, k=len(chunks))
        retrieved = [text for score, text in results]
        rank = retrieved.index(chunks[correct_id]) + 1
        total_rr += 1/rank
        print(f"rank {rank:2d} ← {question}")
    print(f"MRR = {total_rr/len(eval_set)}")


def judge(context, answer_text):
    from generate import generate
    judge_prompt = f"""You are evaluating whether an answer is faithful to a context.
An answer is faithful ONLY if every claim it makes is supported by the context.

Context:
{context}

Answer:
{answer_text}

Is every claim in the answer supported by the context above?
Reply with exactly one word: FAITHFUL or UNFAITHFUL."""
    return generate(judge_prompt).strip()


def recall_at_k_chroma(eval_set, k=3):
    hits = 0
    for question, correct_id in eval_set:
        retrieved = retrieve_chroma(question, k)
        if chunks[correct_id] in retrieved:
            hits += 1
        else:
            print(f"chroma miss: {question}")
    print(f"chroma recall@{k} = {hits/len(eval_set)}")


def recall_at_k_rerank(eval_set, k=3, initial_k=10):
    hits = 0
    for question, correct_id in eval_set:
        retrieved = [doc for score, doc in rerank_retrieve(question, k, initial_k)]
        if chunks[correct_id] in retrieved:
            hits += 1
        else:
            print(f"rerank miss: {question}")
    print(f"rerank recall@{k} = {hits/len(eval_set)}")


# run the eval suite when this file is executed directly
if __name__ == "__main__":
    print("=== baseline (cosine) ===")
    recall_at_k(eval_set)
    mrr(eval_set)
    print("=== chroma ===")
    recall_at_k_chroma(eval_set)
    print("=== rerank ===")
    recall_at_k_rerank(eval_set)