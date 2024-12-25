import os
import time

from typing import List

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from loguru import logger
from contextlib import asynccontextmanager

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, CollectorRegistry, make_asgi_app
from prometheus_client.multiprocess import MultiProcessCollector

from models import Embedder


# Remove all handlers associated with the root logger object
logger.remove()
# Configure the logger
logger.add(
    "./logs/process.log",
    rotation="5 GiB",
    format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {name}:{function}:{line} - {message}"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    global counter
    global error_counter
    global histogram
    
    model_id = os.getenv("EMBEDDER_MODEL_ID")
    device = os.getenv("EMBEDDER_DEVICE")

    # Initialize the embedder
    print("Initializing the Instructed LLM-based embedder...")
    embedder = Embedder(model_id=model_id, device=device, cache_dir="./cache")
    logger.info(f"Embedder initialized with model ID {model_id} and device {device}")

    # Initialize the Prometheus metrics
    logger.info("Initializing Prometheus metrics...")
    counter = Counter("embedding_requests", "Number of embedding requests", ("method", "endpoint"))
    error_counter = Counter("embedding_errors_requests", "Number of errors", ("method", "endpoint"))
    histogram = Histogram("embedding_request_response_time_seconds", "Embedding request response time in seconds", ("method", "endpoint"), unit="seconds")

    logger.info("Model serving service started")

    # Clean up the embedder
    yield
    del embedder
    del counter
    del error_counter
    del histogram

# Add Prometheus ASGI middleware to expose /metrics for multi-process mode
# Acess for more details: https://prometheus.github.io/client_python/exporting/http/fastapi-gunicorn/
def make_metrics_app():
    registry = CollectorRegistry()
    MultiProcessCollector(registry)
    return make_asgi_app(registry=registry)

app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_metrics_app())

# Start instrumenting the FastAPI app
instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)

@app.get("/get-device")
async def get_device():
    """Get the device used by the embedder."""
    return {"device": str(embedder.model.device)}

@app.post("/embed-query")
async def encode_query(request: Request, query: List[str]):
    """Encode a list of queries into query embeddings."""
    # Record the start time
    start_time = time.time()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Received texts to encode: {query}")

    try:
        # Get the query embedding
        query_embeddings = embedder.embed_query(query)

        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        histogram.labels(method="post", endpoint="/embed-query").observe(elapsed_time)
        counter.labels(method="post", endpoint="/embed-query").inc()

        return {"query_embeddings": query_embeddings}
    except Exception as e:
        logger.error(f"Error encoding query: {e}")
        error_counter.labels(method="post", endpoint="/embed-query").inc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/embed-docs")
async def encode_docs(request: Request, docs: List[str]):
    """Encode a list of documents into document embeddings."""
    # Record the start time
    start_time = time.time()

    if not docs:
        raise HTTPException(status_code=400, detail="Docs cannot be empty")

    logger.info(f"Received docs to encode: {docs}")

    try:
        # Get the document embeddings
        doc_embeddings = embedder.embed_doc(docs)

        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        histogram.labels(method="post", endpoint="/embed-docs").observe(elapsed_time)
        counter.labels(method="post", endpoint="/embed-docs").inc()

        return {"doc_embeddings": doc_embeddings}
    except Exception as e:
        logger.error(f"Error encoding docs: {e}")
        error_counter.labels(method="post", endpoint="/embed-docs").inc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
    