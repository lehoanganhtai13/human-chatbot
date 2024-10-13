import os
import sys
from typing import List, Dict
from dotenv import load_dotenv
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import torch

# Add the directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models import LLM

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm

    load_dotenv("../.env")
    
    # Initialize models
    print('Initializing the LLM...')

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    cache_dir = "../cache"
    HF_token = os.getenv("HF_TOKEN") # See https://huggingface.co/docs/hub/security-tokens for more information

    llm = LLM(HF_token, cache_dir, device)

    # Clean up the model
    yield
    del llm

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(request: Request, message: List[Dict[str, str]], max_new_tokens: int = 256):
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Generate a response
    response = llm.generate(message, max_new_tokens)
    return {"response": response}

@app.post("/stream")
async def stream(request: Request, message: List[Dict[str, str]], max_new_tokens: int = 256):
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Generate a streaming response
    streamer = llm.streaming(message, max_new_tokens)
    return StreamingResponse(streamer)

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("llm-app:app", host="0.0.0.0", port=8001)
