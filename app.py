from fastapi import FastAPI, Query
from typing import List
import asyncio

from pipeline.rag_pipeline import RAG_pipeline_async

app = FastAPI(title="Simple RAG Retriever")

@app.get("/retrieve", summary="Retrieve RAG results")
async def retrieve_rag(query: str = Query(..., description="Your search query")):
    try:
        results = await RAG_pipeline_async(query)
        return {"query": query, "results": results}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health():
    return {"status": "ok"}