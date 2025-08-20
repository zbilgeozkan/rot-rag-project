from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .query_faiss import FAISSQuery
from .llm_wrapper import generate_answer

app = FastAPI()
faiss_query = FAISSQuery()

class RAGRequest(BaseModel):
    question: str
    top_k: int = 5

class RAGResponse(BaseModel):
    answer: str
    citations: List[str]
    passages: List[str]

@app.post("/rag/ask", response_model=RAGResponse)
def ask_rag(req: RAGRequest):
    # FAISS Retrieval
    results = faiss_query.query(req.question, top_k=req.top_k)
    passages = [r["text"] for r in results]
    citations = [r["source"] for r in results]

    # LLM Generating Response
    answer = generate_answer(req.question, passages)

    return RAGResponse(answer=answer, citations=citations, passages=passages)