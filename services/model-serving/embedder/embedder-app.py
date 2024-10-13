import os
import sys
from typing import List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, HTTPException
import torch

# Add the directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models import Embedder

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder

    # Initialize models
    print('Initializing the embedder...')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cache_dir = "../cache"
    embedder = Embedder(cache_dir, device)

    # Clean up the model
    yield
    del embedder

app = FastAPI(lifespan=lifespan)

@app.post("/embed-query")
async def encode_query(request: Request, query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Get the query embedding
    query_embedding = embedder.embed_query([query])[0]

    return {"query_embedding": query_embedding}

@app.post("/embed-docs")
async def encode_docs(request: Request, docs: List[str]):
    if not docs:
        raise HTTPException(status_code=400, detail="Docs cannot be empty")

    # Get the document embeddings
    doc_embeddings = embedder.embed_doc(docs)

    return {"doc_embeddings": doc_embeddings}

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("embedder-app:app", host="0.0.0.0", port=8000)
