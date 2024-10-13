from typing import Any, List, Dict
import requests

from deep_translator import GoogleTranslator
from llama_index.core import get_response_synthesizer
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import NodeWithScore
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata, ChatMessage
from llama_index.core.llms.callbacks import llm_completion_callback

from prompt import qa_prompt

# See: https://docs.llamaindex.ai/en/stable/api_reference/storage/chat_store/redis/
class ChatStore():
    def __init__(self, host: str, port: int, db: int, username: str, password: str, live_time_seconds: int = 86400, max_messages_pairs: int = 5):
        self.chat_client = RedisChatStore(redis_url=f"redis://{username}:{password}@{host}:{port}/{db}", ttl=live_time_seconds)
        self.max_messages_pairs = max_messages_pairs

    def chat_messages_to_dict(self, messages, en_translate=False):
        """Convert chat messages to LLM chat format with translation."""
        en_translator = GoogleTranslator(source="auto", target="en")
        messages_dict = []
        for message in messages:
            content = message.content if not en_translate else en_translator.translate(message.content).capitalize()
            messages_dict.append({"role": message.role.value, "content": content})
        return messages_dict

    def trim_messages(self, key):
        """Trim the messages if the number of messages exceeds the limit."""

        # We store messages in pairs of user and assistant messages
        max_messages = self.max_messages_pairs * 2

        # Get all messages
        messages = self.chat_client.get_messages(key)

        keep_messages = messages[-max_messages:]
        self.chat_client.set_messages(key, keep_messages)
        remove_messages = messages[:-max_messages]

        return remove_messages

    async def async_trim_messages(self, key):
        """Trim the messages if the number of messages exceeds the limit asynchronously."""

        # We store messages in pairs of user and assistant messages
        max_messages = self.max_messages_pairs * 2

        # Get all messages
        messages = await self.chat_client.aget_messages(key)

        keep_messages = messages[-max_messages:]
        await self.chat_client.aset_messages(key, keep_messages)
        remove_messages = messages[:-max_messages]

        return remove_messages

    def add_message_pair(self, key: str, query: str, response: str):
        """Add a query-response message pair to the chat store."""
        message_pair = [
            ChatMessage(role="user", content=query),
            ChatMessage(role="assistant", content=response),
        ]
        for message in message_pair:
            self.chat_client.add_message(key, message)

        # Trim the messages if the number of messages exceeds the limit
        removed_messages = self.trim_messages(key)
        if removed_messages:
            return self.chat_messages_to_dict(removed_messages)

        return None

    async def async_add_message_pair(self, key: str, query: str, response: str):
        """Add a query-response message pair to the chat store asynchronously."""
        message_pair = [
            ChatMessage(role="user", content=query),
            ChatMessage(role="assistant", content=response),
        ]
        for message in message_pair:
            await self.chat_client.async_add_message(key, message)

        # Trim the messages if the number of messages exceeds the limit
        removed_messages = await self.async_trim_messages(key)
        if removed_messages:
            return self.chat_messages_to_dict(removed_messages)

        return None

    def get_chat_history(self, key: str, en_translate=False):
        """Get the chat history from the chat store."""
        messages = self.chat_client.get_messages(key)
        return self.chat_messages_to_dict(messages, en_translate)

    async def async_get_chat_history(self, key: str, en_translate=False):
        """Get the chat history from the chat store asynchronously."""
        messages = await self.chat_client.aget_messages(key)
        return self.chat_messages_to_dict(messages, en_translate)

    def clear_messages(self, key: str):
        """Clear the messages of the user from the chat store."""
        self.chat_client.delete_messages(key)
        return None

    async def async_clear_messages(self, key: str):
        """Clear the messages of the user from the chat store asynchronously."""
        await self.chat_client.adelete_messages(key)
        return None

# See: https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#example-using-a-custom-llm-model-advanced
class LLMCore(CustomLLM):
    uri: str = "http://localhost:8001"
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_new_tokens: int = 256
    chat_store: ChatStore = None

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model_id,
            max_new_tokens=self.max_new_tokens,
            uri=self.uri
        )

    def format_message(self, text: str) -> List[Dict]:
        if not text:
            raise ValueError("No text provided to format.")

        text = text.strip()

        text_split = text.split("|")
        full_prompt_query_text = text_split[0] + text_split[2]
        user_id = text_split[1]

        # Get chat history
        chat_history = self.chat_store.get_chat_history(user_id, en_translate=True)

        prompt = '\n'.join(full_prompt_query_text.split("\n")[:-1])
        query = text.split("\n")[-1]

        return [
            {"role": "system", "content": prompt},
            *chat_history,
            {"role": "user", "content": query},
        ]

    def call_server(self, messages: List[Dict], generate_type: str) -> str:
        """Call the LLM server."""
        if not messages:
            raise ValueError("No messages provided to the LLM server.")

        if generate_type == "generate":
            try:
                url = f"{self.uri}/generate?max_new_tokens={self.max_new_tokens}"
                response = requests.post(
                    url,
                    headers = {
                        'accept': 'application/json',
                        'Content-Type': 'application/json'
                    },
                    json = messages
                )
                return response.json()["response"]["content"]
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Error calling the LLM server at endpoint /generate: {e}")
        elif generate_type == "streaming":
            try:
                url = f"{self.uri}/stream?max_new_tokens={self.max_new_tokens}"
                response = requests.post(
                    url,
                    headers = {
                        'accept': 'application/json',
                        'Content-Type': 'application/json'
                    },
                    json = messages,
                    stream=True
                )
                return response
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Error calling the LLM server at endpoint /streaming: {e}")

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = self.format_message(prompt)
        generated_text = self.call_server(messages, "generate")
        return CompletionResponse(text=generated_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        messages = self.format_message(prompt)
        streamer = self.call_server(messages, "streaming")
        response = ""
        for token in streamer.iter_content(chunk_size=1, decode_unicode=True):
            if isinstance(token, bytes):
                token = token.decode('utf-8')
            response += token
            yield CompletionResponse(text=response, delta=token)

# See: https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/response_synthesizers/
class  Generator():
    def __init__(self, llm: LLMCore, chat_store: ChatStore, max_new_tokens: int = 256, streaming: bool = False):
        llm.chat_store = chat_store
        self.stream_generator = get_response_synthesizer(
            llm=llm,
            text_qa_template=qa_prompt.partial_format(max_num_tokens=max_new_tokens),
            streaming=streaming
        )
        self.completion_generator = get_response_synthesizer(
            llm=llm,
            text_qa_template=qa_prompt.partial_format(max_num_tokens=max_new_tokens),
            streaming=False
        )
        self.streaming = streaming

    def generate(self, query: str, nodes: List[NodeWithScore]):
        if self.streaming:
            response = self.stream_generator.synthesize(query, nodes)
            return response.response_gen
        
        response = self.completion_generator.synthesize(query, nodes)
        return response.response
