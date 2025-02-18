import requests
import threading
import tiktoken
from typing import Any, List, Dict, Generator, Optional

from transformers import AutoTokenizer
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI

from chatbot.core.chat_stores import CacheChatStore
from chatbot.core.model_clients.exceptions import CallServerLLMError, FormatChatMessageError, InitWebsocketOpenAIError
from chatbot.OpenAI.websocket_client import OpenAIWebSocketClient, OPENAI_WEBSOCKET_URI


PROMPT_DELIMITER = """
---------------------
##### REAL DATA #####
---------------------
"""

CHAT_PROMPT_DELIMITER = "<|>"


# See: https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#example-using-a-custom-llm-model-advanced
class LLMCore(CustomLLM):
    uri: str = "http://localhost:8000"
    _model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_new_tokens: int = 256
    chat_store: Optional[CacheChatStore] = None
    is_chat: bool = False
    use_openai: bool = False
    openai_base_url: str = "https://api.openai.com"
    OPENAI_API_KEY: str = "EMPTY"
    temperature: float = 0.7
    cache_dir: str = "/app/cache"
    count_token: bool = False
    use_history: bool = False
    user_id: Optional[str] = None
    assistant_id: Optional[str] = None
    websocket_client: Optional[OpenAIWebSocketClient] = None
    connection_thread: Optional[threading.Thread] = None
    encoding: Optional[tiktoken.Encoding] = None
    tokenizer: Optional[AutoTokenizer] = None
    chat_delimiter: str = "<|>"
    function_name: str = None

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        use_websocket: bool = False,
        count_token: bool = False,
        chat_prompt_delimiter: str = None,
        **kwargs: Any
    ) -> None:
        """
        Custom LLM class to call API to LLM server (either local, OpenAI or OpenAI-compatible 3rd party server).
            
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
            - cache_dir (str): The cache directory for storing tokenizer.
            - count_token (bool): Whether to count the number of tokens in the text.
            - use_history (bool): Whether to use chat history with prompt when calling the LLM server.
            - user_id (str): The user ID for calling history.
            - assistant_id (str): The assistant ID for calling history.
            - websocket_client (OpenAIWebSocketClient): The websocket client for streaming.
            - connection_thread (threading.Thread): The thread for websocket connection.
            - encoding (tiktoken.Encoding): The encoding for tokenization (OpenAI).
            - tokenizer (AutoTokenizer): The tokenizer for tokenization (Hugging Face).
            - chat_delimiter (str): The delimiter for response mixed prompt containing the original prompt and query. 
        """
        super().__init__(**kwargs)

        # Set the model ID
        self._model_id = model_id
        
        # Set the delimiter for splitting the input chat prompt
        self.chat_delimiter = chat_prompt_delimiter or CHAT_PROMPT_DELIMITER

        if count_token:
            self.setup_token_counter()

        if self.use_openai and use_websocket:
            self.setup_websocket()

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self._model_id,
            max_new_tokens=self.max_new_tokens,
            uri=self.uri,
            use_openai=self.use_openai
        )
    
    def setup_token_counter(self) -> None:
        """Initialize token counter based on model type"""
        if self.use_openai and "gpt" in self._model_id:
            self.encoding = tiktoken.encoding_for_model(self._model_id)
        else:
            if self.use_openai:
                print("Invalid model ID for OpenAI. Using tokenizer from Hugging Face.")
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_id, cache_dir=self.cache_dir)

    def setup_websocket(self) -> None:
        """Initialize and start websocket connection"""
        if "gpt-4o" not in self._model_id:
            raise InitWebsocketOpenAIError("Realtime API is only supported for GPT-4o family models.")
        
        try:
            realtime_model_id = self._model_id + "-realtime-preview-2024-12-17"
            websocket_uri = OPENAI_WEBSOCKET_URI + realtime_model_id
            self.websocket_client = OpenAIWebSocketClient(
                api_key=self.OPENAI_API_KEY,
                uri=websocket_uri,
                headers={
                    "Authorization": f"Bearer {self.OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
            self.connection_thread = threading.Thread(target=self.websocket_client.connect)
            self.connection_thread.start()
            print("Started connection to OpenAI Websocket server.")
        except Exception as e:
            raise InitWebsocketOpenAIError(f"Error initializing websocket connection: {e}")

    def set_function_name(self, name: str) -> None:
        """Set the function name for calling LLM server history."""
        self.function_name = name

    def clear_function_name(self) -> None:
        """Clear the function name for calling LLM server history."""
        self.function_name = None

    def clean_call_history(self) -> None:
        """Clean the calling LLM server history."""
        if self.use_history and self.function_name:
            self.chat_store.clear_messages(f"{self.function_name}_{self.user_id}_{self.assistant_id}")

    def text_generator(self, streamer) -> Generator[str, None, None]:
        """
        Generate text from the streamer object of the OpenAI websocket client.
        
        Args:
            streamer: The streamer to generate text from.

        Yields:
            Generator[str, None, None]: The generated text from the streamer.
        """
        for output in streamer:
            if output == "[END]":
                continue
            yield output

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text (str): The text to count tokens.

        Returns:
            int: The number of tokens in the text.
        """
        if self.use_openai:
            if self.encoding is not None:
                tokens = self.encoding.encode(text)
            else:
                tokens = self.tokenizer.tokenize(text)
        else:
            tokens = self.tokenizer.tokenize(text)
        return len(tokens)

    def format_chat_message(self, text: str) -> List[Dict]:
        """
        Format the chat message with history to be sent to the LLM server.
        
        Args:
            text (str): The text to format.

        Returns:
            List[Dict]: The formatted chat message with history.
        """
        if not text:
            raise FormatChatMessageError("No text provided to format.")

        text = text.strip()

        try:
            text_split = text.split(self.chat_delimiter)

            # Get chat history corresponding with original language to keep the final response in the same language
            chat_history = self.chat_store.get_chat_history(f"original_{self.user_id}_{self.assistant_id}", en_translate=False)

            prompt = text_split[0].strip()
            query = text_split[1].strip()

            return [
                {"role": "system", "content": prompt},
                *chat_history,
                {"role": "user", "content": query},
            ]
        except Exception as e:
            raise FormatChatMessageError(f"Error formatting chat message: {e}")
            
    def format_call_message(self, prompt: str, name: Optional[str] = None) -> str:
        """
        Format the call message to the LLM server.
        
        Args:
            prompt (str): The prompt to send to the LLM server.
            name (str): The name of the calling function.
        Returns:
            str: The formatted call message.
        """

        chat_history = []
        if self.use_history and name:
            chat_history = self.chat_store.get_chat_history(f"{name}_{self.user_id}_{self.assistant_id}", en_translate=False)

        if PROMPT_DELIMITER in prompt:
            system_prompt = prompt.split(PROMPT_DELIMITER)[0]
            query = prompt.split(PROMPT_DELIMITER)[1]
            messages = [
                {"role": "system", "content": system_prompt},
                *chat_history,
                {"role": "user", "content": query}
            ]
        else:
            messages = [
                *chat_history,
                {"role": "user", "content": prompt}
            ]

        return messages

    def call_server(self, messages: List[Dict], generate_type: str) -> str:
        """
        Call the LLM server.
        
        Args:
            messages (List[Dict]): The messages to send to the LLM server.
            generate_type (str): The type of generation to perform.

        Returns:
            str: The generated text from the LLM server.
        """
        if not messages:
            raise CallServerLLMError("No messages provided to the LLM server.")

        url = f"{self.uri}/v1"
        if self.use_openai:
            url = f"{self.openai_base_url}/v1"

        if self.count_token:
            try:
                if self.use_openai:
                    # Count the total tokens in the messages
                    total_tokens = 0
                    for message in messages:
                        total_tokens += self.count_tokens(message["content"])
                    print(f"Total tokens send to OpenAI: {total_tokens}")
                else:
                    total_tokens = 0
                    for message in messages:
                        total_tokens += self.count_tokens(message["content"])
                    print(f"Total tokens send to LLM: {total_tokens}")
            except Exception as e:
                raise CallServerLLMError(f"Error counting tokens: {e}")
                
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
                raise CallServerLLMError(f"Error calling the LLM server at endpoint /generate: {e}")
        elif generate_type == "streaming":
            try:
                if self.use_openai and self.websocket_client is not None:
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
                raise CallServerLLMError(f"Error calling the LLM server at endpoint /streaming: {e}")

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = ""

        # Format the message based on the chat or call
        if not self.is_chat:
            messages= self.format_call_message(prompt, self.function_name)
        else:
            messages = self.format_chat_message(prompt)

        # Call the LLM server to generate text
        generated_text = self.call_server(messages, "generate")

        # Store system prompt to the calling history
        if self.use_history and self.function_name and PROMPT_DELIMITER in prompt:
            self.chat_store.add_system_message(
                key=f"{self.function_name}_{self.user_id}_{self.assistant_id}",
                message=messages[0]["content"]
            )

        # Store user query and assistant response to the calling history
        if self.use_history and self.function_name:
            self.chat_store.add_message_pair(
                key=f"{self.function_name}_{self.user_id}_{self.assistant_id}",
                query=messages[-1]["content"],
                response=generated_text
            )

        return CompletionResponse(text=generated_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        messages = ""

        # Format the message based on the chat or call
        if not self.is_chat:
            messages= self.format_call_message(prompt, self.function_name)
        else:
            messages = self.format_chat_message(prompt)

        # Call the LLM server to generate text
        streamer = self.call_server(messages, "streaming")

        # Return the generator for streaming
        response = ""
        for output in streamer:
            token = output
            if not self.websocket_client:
                token = output.dict()["choices"][0]["delta"]["content"]
            if token is not None:
                for char in token:
                    response += char
                    yield CompletionResponse(text=response, delta=char)

        # Store system prompt to the calling history
        if self.use_history and self.function_name and PROMPT_DELIMITER in prompt:
            self.chat_store.add_system_message(
                key=f"{self.function_name}_{self.user_id}_{self.assistant_id}",
                message=messages[0]["content"]
            )

        # Store user query and assistant response to the calling history
        if self.use_history and self.function_name:
            self.chat_store.add_message_pair(
                key=f"{self.function_name}_{self.user_id}_{self.assistant_id}",
                query=messages[-1]["content"],
                response=response
            )
