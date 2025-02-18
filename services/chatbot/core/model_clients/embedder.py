import requests
import tiktoken
from typing import Any, List

from transformers import AutoTokenizer
from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI

from chatbot.core.model_clients.exceptions import CallServerEmbedderError


# See: https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/#custom-embedding-model
class EmbedderCore(BaseEmbedding):
    """Custom embedding class to call API to Embedder server."""

    uri: str = "http://localhost:8011"
    model_id: str = "dunzhang/stella_en_400M_v5"
    use_openai: bool = False
    OPENAI_API_KEY: str = "EMPTY"
    encoding: tiktoken.Encoding = None
    tokenizer: AutoTokenizer = None
    count_token: bool = False

    def __init__(
        self,
        uri: str = "http://localhost:8011",
        model_id: str = "dunzhang/stella_en_400M_v5",
        use_openai: bool = False,
        OPENAI_API_KEY: str = "EMPTY",
        count_token: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.uri = uri
        self.model_id = model_id
        self.use_openai = use_openai
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.count_token = count_token

        if count_token:
            if self.use_openai:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text (str): The text to count tokens for.

        Returns:
            int: The number of tokens in the text.
        """
        if self.use_openai:
            tokens = self.encoding.encode(text)
        else:
            tokens = self.tokenizer.tokenize(text)
        return len(tokens)

    def _call_API(self, text: List[str], type: str) -> List[List[float]]:
        """
        Call the API to get the corresponding embeddings.
        
        Args:
            text (List[str]): The list of texts to get embeddings for.
            type (str): The type of text to get embeddings for.

        Returns:
            List[List[float]]: The list of embeddings for the texts.
        """
        try:
            if self.use_openai:
                client = OpenAI(api_key=self.OPENAI_API_KEY)
                response = client.embeddings.create(
                    input = text, model=self.model_id, dimensions=1024
                )
                return [res.embedding for res in response.data]

            if type == "query":
                response = requests.post(
                    f'{self.uri}/embed-query',
                    headers={
                        'accept': 'application/json',
                        'Content-Type': 'application/json'
                    },
                    json=text
                )
                return response.json()["query_embeddings"]
            else:
                response = requests.post(
                    f'{self.uri}/embed-docs',
                    headers={
                        'accept': 'application/json',
                        'Content-Type': 'application/json'
                    },
                    json=text
                )
                return response.json()["doc_embeddings"]
        except Exception as e:
            raise CallServerEmbedderError(f"Error calling Embedder server: {e!s}")
        
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get the embedding for the query.

        Args:
            query (str): The query to get the embedding for.

        Returns:
            List[float]: The embedding for the query.
        """
        if self.use_openai and self.count_token:
            print("Number of tokens in query sent to OpenAI:", self.count_tokens(query))
        embeddings = self._call_API([query], "query")
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for the text.

        Args:
            text (str): The text to get the embedding for.

        Returns:
            List[float]: The embedding for the text.
        """
        if self.use_openai and self.count_token:
            print("Number of tokens in text sent to OpenAI:", self.count_tokens(text))
        embeddings = self._call_API([text], "text")
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get the embeddings for the list of texts.

        Args:
            texts (List[str]): The list of texts to get embeddings for.

        Returns:
            List[List[float]]: The list of embeddings for the texts.
        """
        if self.use_openai and self.count_token:
            print("Number of tokens in texts sent to OpenAI:", sum([self.count_tokens(text) for text in texts]))
        embeddings = self._call_API(texts, "text")
        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Asynchronously get the embedding for the query.

        Args:
            query (str): The query to get the embedding for.

        Returns:
            List[float]: The embedding for the query.
        """
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Asynchronously get the embedding for the text.

        Args:
            text (str): The text to get the embedding for.

        Returns:
            List[float]: The embedding for the text.
        """
        return self._get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously get the embeddings for the list of texts.

        Args:
            texts (List[str]): The list of texts to get embeddings for.

        Returns:
            List[List[float]]: The list of embeddings for the texts.
        """
        return self._get_text_embeddings(texts)
    