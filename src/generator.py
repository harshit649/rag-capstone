from generate import generate
from retriever import retrieve
from ingest import chunks


def answer(query):
    results = retrieve(query, chunks, k=3)
    context = "\n\n".join(text for score, text in results)
    prompt = f"""Use ONLY the context below to answer. If the answer isn't there, say you don't know
    Context
    {context}
    Question
    {query}"""
    return generate(prompt)