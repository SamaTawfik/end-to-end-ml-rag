import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.db.vector_db import VectorDBManager
from src.pipeline.predict_pipeline import RAGPredictPipeline

load_dotenv()

app = FastAPI(title="Mini-RAG Application API")

sample_documents = [
    "Sama Tawfik is a computer engineering student graduating in 2027.",
    "The Mini-RAG architecture contains src, api, db, utils, and pipeline folders."
]

vector_db = VectorDBManager()
vector_db.create_vector_store(sample_documents)
retriever = vector_db.get_retriever()

rag_pipeline = RAGPredictPipeline(retriever=retriever)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Mini-RAG API"}

@app.post("/predict", response_model=QueryResponse)
def predict(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    answer = rag_pipeline.predict(request.question)
    return QueryResponse(question=request.question, answer=answer)