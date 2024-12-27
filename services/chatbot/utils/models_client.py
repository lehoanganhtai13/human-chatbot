import requests
import threading
import tiktoken
from typing import Any, List, Dict

from transformers import AutoTokenizer
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI

from chatbot.utils.chat_store import CacheChatStore
from chatbot.OpenAI.websocket_client import OpenAIWebSocketClient, OPENAI_WEBSOCKET_URI


PROMPT_DELIMITER = """
---------------------
##### REAL DATA #####
---------------------
"""


# See: https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/#custom-embedding-model
class EmbedderCore(BaseEmbedding):
    """Custom embedding class to call API to Embedder server."""

    uri: str = "http://localhost:8011"
    model_id: str = "dunzhang/stella_en_400M_v5"

    def __init__(self, uri: str = "http://localhost:8011", model_id: str = "dunzhang/stella_en_400M_v5", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.uri = uri
        self.model_id = model_id

    def _call_API(self, text: List[str], type: str) -> List[List[float]]:
        """Call the API to get the corresponding embeddings."""
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
        
    def _get_query_embedding(self, query: str) -> List[float]:
            embeddings = self._call_API([query], "query")
            return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._call_API([text], "text")
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._call_API(texts, "text")
        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)
    

# See: https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#example-using-a-custom-llm-model-advanced
class LLMCore(CustomLLM):
    """Custom LLM class to call API to LLM server (either local or OpenAI).
        
    Attributes:
        - uri (str): The URI of the LLM server.
        - model_id (str): The model ID of the LLM server.
        - max_new_tokens (int): The maximum number of tokens to generate.
        - chat_store (CacheChatStore): The chat store to store chat history.
        - is_chat (bool): Whether the LLM is used for conversation chat or just for generating response based on instruction prompt.
        - use_openai (bool): Whether to use OpenAI API.
        - OPENAI_API_KEY (str): The OpenAI API key.
        - temperature (float): The temperature for sampling.
        - use_websocket (bool): Whether to use websocket for streaming.
    """

    uri: str = "http://localhost:8013"
    _model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_new_tokens: int = 256
    chat_store: CacheChatStore = None
    is_chat: bool = False
    use_openai: bool = False
    OPENAI_API_KEY: str = "EMPTY"
    temperature: float = 0.7
    websocket_client: OpenAIWebSocketClient = None
    connection_thread: threading.Thread = None

    def __init__(self, uri: str = "http://localhost:8013", model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens: int = 256, chat_store: CacheChatStore = None, is_chat: bool = False, use_openai: bool = False, OPENAI_API_KEY: str = "EMPTY", temperature: float = 0.7, use_websocket: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.uri = uri
        self._model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.chat_store = chat_store
        self.is_chat = is_chat
        self.use_openai = use_openai
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.temperature = temperature

        if self.use_openai and use_websocket:
            self.websocket_client = OpenAIWebSocketClient(
                api_key=self.OPENAI_API_KEY,
                uri=OPENAI_WEBSOCKET_URI,
                headers={
                    "Authorization": f"Bearer {self.OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
            # Start the connection in a separate thread
            self.connection_thread = threading.Thread(target=self.websocket_client.connect)
            self.connection_thread.start()

            print("Started connection to OpenAI Websocket server.")

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self._model_id,
            max_new_tokens=self.max_new_tokens,
            uri=self.uri,
            use_openai=self.use_openai
        )

    def text_generator(self, streamer):
        """Generate text from the streamer."""
        for output in streamer:
            if output == "[END]":
                continue
            yield output

    def count_tokens(self, text: str, model_id: str) -> int:
        """Count the number of tokens in the text."""
        if self.use_openai:
            encoding = tiktoken.encoding_for_model(model_id)
            tokens = encoding.encode(text)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/app/cache")
            tokens = tokenizer.tokenize(text)
        return len(tokens)

    def format_message(self, text: str) -> List[Dict]:
        """Format the chat message with history to be sent to the LLM server."""
        if not text:
            raise ValueError("No text provided to format.")

        text = text.strip()

        text_split = text.split("|")
        full_prompt_query_text = text_split[0] + text_split[3]
        user_id = text_split[1]
        assistant_id = text_split[2]

        # Get chat history corresponding with original language
        chat_history = self.chat_store.get_chat_history(f"original_{user_id}_{assistant_id}", en_translate=False) # TODO: Optimize to translate faster

        prompt = '\n'.join(full_prompt_query_text.split("\n")[:-1])
        # Remove the initial introduction sentence for the retrieved context from the query
        if "Here are some facts extracted from the provided text:" in prompt:
            prompt = prompt.replace("Here are some facts extracted from the provided text:\n", "")

        query = text.split("\n")[-1].split("|")[-1]

        return [
            {"role": "system", "content": prompt},
            *chat_history,
            {"role": "user", "content": query},
        ]

    def call_server(self, messages: List[Dict], generate_type: str) -> str:
        """Call the LLM server."""
        if not messages:
            raise ValueError("No messages provided to the LLM server.")

        url = f"{self.uri}/v1"
        if self.use_openai:
            url = "https://api.openai.com/v1"

            # Count the total tokens in the messages
            total_tokens = 0
            for message in messages:
                total_tokens += self.count_tokens(message["content"], self._model_id)
            print(f"Total tokens send to OpenAI: {total_tokens}")
        else:
            total_tokens = 0
            for message in messages:
                total_tokens += self.count_tokens(message["content"], self._model_id)
            print(f"Total tokens send to LLM: {total_tokens}")
            
        client = OpenAI(api_key=self.OPENAI_API_KEY, base_url=url)
        if generate_type == "generate":
            try:
                response = client.chat.completions.create(
                    model=self._model_id,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    stream=False,
                )
                response_text = response.model_dump()["choices"][0]["message"]["content"]
                return response_text.replace("{{", "{").replace("}}", "}")
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Error calling the LLM server at endpoint /generate: {e}")
        elif generate_type == "streaming":
            try:
                if self.use_openai:
                    data = [
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "system",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": messages[0]["content"],
                                    }
                                ]
                            }
                        },
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": messages[-1]["content"],
                                    }
                                ]
                            }
                        },
                        {
                            "type": "response.create",
                            "response": {"modalities": ["text"]}
                        }
                    ]
                    self.websocket_client.send_message(data)
                    return self.text_generator(self.websocket_client.token_generator())
                else:
                    response = client.chat.completions.create(
                        model=self._model_id,
                        messages=messages,
                        max_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        stream=True
                    )
                    return response
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Error calling the LLM server at endpoint /streaming: {e}")

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = ""
        if not self.is_chat:
            if PROMPT_DELIMITER in prompt:
                print("Prompt delimiter found")
                system_prompt = prompt.split(PROMPT_DELIMITER)[0]
                query = prompt.split(PROMPT_DELIMITER)[1]
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
        else:
            messages = self.format_message(prompt)
        generated_text = self.call_server(messages, "generate")
        return CompletionResponse(text=generated_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        messages = ""
        if not self.is_chat:
            if PROMPT_DELIMITER in prompt:
                print("Prompt delimiter found")
                system_prompt = prompt.split(PROMPT_DELIMITER)[0]
                query = prompt.split(PROMPT_DELIMITER)[1]
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
        else:
            messages = self.format_message(prompt)
        streamer = self.call_server(messages, "streaming")
        response = ""
        for output in streamer:
            token = output
            if not self.use_openai:
                token = output.dict()["choices"][0]["delta"]["content"]
            if token is not None:
                for char in token:
                    response += char
                    yield CompletionResponse(text=response, delta=char)
