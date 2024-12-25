from typing import List

import torch
from FlagEmbedding import FlagLLMReranker
from sentence_transformers import SentenceTransformer
    
    
# See: https://huggingface.co/dunzhang/stella_en_400M_v5
class Embedder():
    def __init__(self, model_id: str = "dunzhang/stella_en_400M_v5", cache_dir: str = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device != "cpu" else "cpu"

        config_kwargs = {
            "use_memory_efficient_attention": False, "unpad_inputs": False
        } if self.device == "cpu" else {}

        self.model = SentenceTransformer(
            model_id,
            device=self.device,
            cache_folder=cache_dir,
            trust_remote_code=True,
            config_kwargs=config_kwargs
        )
        self.query_prompt_type = "s2p_query"

    def embed_query(self, queries: List[str]) -> list:
        query_embeddings = self.model.encode(
            queries, prompt_name=self.query_prompt_type
        )
        embeddings_list = query_embeddings.tolist()
        del query_embeddings, queries
        torch.cuda.empty_cache()
        return embeddings_list

    def embed_doc(self, docs: List[str]) -> list:
        doc_embeddings = self.model.encode(docs)
        embeddings_list = doc_embeddings.tolist()
        del doc_embeddings, docs
        torch.cuda.empty_cache()
        return embeddings_list


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
