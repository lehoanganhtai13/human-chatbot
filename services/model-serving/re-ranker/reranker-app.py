import os
import sys
from typing import List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, HTTPException
import torch

# Add the directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models import ReRanker

@asynccontextmanager
async def lifespan(app: FastAPI):
    global re_ranker

    # Initialize models
    print('Initializing the re-ranker...')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cache_dir = "../cache"
    re_ranker = ReRanker(cache_dir, device)

    # Clean up the model
    yield
    del re_ranker

app = FastAPI(lifespan=lifespan)

@app.post("/re-ranking")
async def re_ranking(request: Request, pairs: List[List[str]]):
    if not pairs:
        raise HTTPException(status_code=400, detail="No data provided")

    try:
        scores = re_ranker.rerank(pairs)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"sorted_indices": sorted_indices}

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("reranker-app:app", host="0.0.0.0", port=8002)
