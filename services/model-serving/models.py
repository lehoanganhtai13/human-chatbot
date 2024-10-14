import os
from typing import List, Dict
from dotenv import load_dotenv

from threading import Thread

import torch
import transformers
from FlagEmbedding import FlagLLMReranker
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, TextIteratorStreamer


load_dotenv(".env")

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    },
    "chat": {
        "query": "Embed this dialogue to find useful historical dialogues: ",
        "key": "Embed this historical dialogue for retrieval: ",
    },
    "lrlm": {
        "query": "Embed this text chunk for finding useful historical chunks: ",
        "key": "Embed this historical text chunk for retrieval: ",
    },
    "tool": {
        "query": "Transform this user request for fetching helpful tool descriptions: ",
        "key": "Transform this tool description for retrieval: "
    },
    "convsearch": {
        "query": "Encode this query and context for searching relevant passages: ",
        "key": "Encode this passage for retrieval: ",
    },
}

# See: https://arxiv.org/abs/2407.01219 and https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder
class Embedder():
    def __init__(self, cache_dir: str, device: str = "cuda"):
        model_id = "BAAI/llm-embedder"
        self.model = SentenceTransformer(
            model_id,
            device=device,
            cache_folder=cache_dir
        )
        self.instruction = INSTRUCTIONS["qa"]

    def embed_query(self, queries: List[str]) -> torch.Tensor:
        queries = [self.instruction["query"] + query for query in queries]
        return self.model.encode(queries).tolist()

    def embed_doc(self, docs: List[str]) -> torch.Tensor:
        docs = [self.instruction["key"] + doc for doc in docs]
        return self.model.encode(docs).tolist()

# See: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
class LLM():
    def __init__(self, HF_token: str, cache_dir: str, device: str = "cuda"):
        if device == "cpu":
            torch_type = torch.bfloat16
            model_id = "meta-llama/Llama-3.2-3B-Instruct" # This will take about 3 minutes to generate using CPU
        else:
            torch_type = torch.float16
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        self.model = transformers.pipeline(
            "text-generation",
            model=model_id,
            device=device,
            model_kwargs={"torch_dtype": torch_type, "cache_dir": cache_dir},
            token=HF_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        return self.model(
            messages, max_new_tokens=max_new_tokens, 
            pad_token_id=self.model.tokenizer.eos_token_id
        )[0]["generated_text"][-1]

    def streaming(self, messages: List[Dict[str, str]], max_new_tokens: int):
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            text_inputs=messages, 
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )
        thread = Thread(target=self.model, kwargs=generation_kwargs)
        thread.start()
        return streamer

# See: https://huggingface.co/BAAI/bge-reranker-v2-gemma
class ReRanker():
    def __init__(self, cache_dir: str, device: str = "cuda"):
        model_id = "BAAI/bge-reranker-v2-gemma"
        self.model = FlagLLMReranker(
            model_id,
            use_fp16=True, # Setting use_fp16 to True speeds up computation with a slight performance degradation
            device=device,
            cache_dir=cache_dir
        )

    def rerank(self, pairs: List[List[str]]) -> List[int]:
        return self.model.compute_score(pairs)
