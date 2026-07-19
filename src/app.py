from fastapi import FastAPI
from pydantic import BaseModel
from generator import answer
from retriever import rerank_retrieve

app = FastAPI(title="RAG Capstone API")


class Query(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "ok", "service": "rag-capstone"}


@app.post("/query")
def query_rag(q: Query):
    # retrieve best chunks via reranking (the 1.0 recall path), then generate
    contexts = [doc for score, doc in rerank_retrieve(q.question, k=3)]
    response = answer(q.question)
    return {
        "question": q.question,
        "answer": response,
        "contexts": contexts,
    }