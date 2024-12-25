from typing import List, Dict

from pymongo import AsyncMongoClient, MongoClient
from datetime import datetime, timezone

from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, CustomLLM
from llama_index.storage.chat_store.redis import RedisChatStore

from chatbot.prompt.graph.summary import SUMMARIZE_PROMPT_TEMPLATE
from chatbot.utils.translator import Translator


class CacheChatStore:
    """A class to store chat messages in a cache."""

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        username: str,
        password: str,
        llm: CustomLLM,
        live_time_seconds: int = 86400,
        max_messages_pairs: int = 5,
    ):
        self.chat_client = RedisChatStore(
            redis_url=f"redis://{username}:{password}@{host}:{port}/{db}",
            ttl=live_time_seconds,
        )
        self.max_messages_pairs = (
            max_messages_pairs + 1
            if max_messages_pairs % 2 != 0
            else max_messages_pairs
        )  # Ensure even number of messages
        self.en_translator = Translator(
            source="auto", target="english", capitalize_sentences=True
        )
        self.llm = llm

    def chat_messages_to_dict(self, messages, en_translate=False):
        """Convert chat messages to LLM chat format with translation."""

        messages_dict = []
        for message in messages:
            content = (
                message.content
                if not en_translate
                else self.en_translator.translate(message.content)
            )
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

    def get_chat_history(self, key: str, en_translate=False) -> List[Dict[str, str]]:
        """Get the chat history from the chat store."""
        messages = self.chat_client.get_messages(key)
        return self.chat_messages_to_dict(messages, en_translate)

    async def async_get_chat_history(
        self, key: str, en_translate=False
    ) -> List[Dict[str, str]]:
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

    def transform_message_pair(
        self, message_pair: List[Dict[str, str]], user_id: str, assistant_id: str, user_cache_id: str
    ):
        """Transform the user and assistant messages to a single memory message."""
        user_message = message_pair[0]["content"]
        assistant_response = message_pair[1]["content"]
        memory_message = f"""{user_id} said to {assistant_id}, "{user_message}" and {assistant_id} replied, "{assistant_response}"."""

        user_messages = []
        assistant_messages = []
        for message in self.chat_client.get_messages(user_cache_id):
            if message.role == "user":
                user_messages.append(message.content)
            else:
                assistant_messages.append(message.content)

        processed_chat_history = []
        for user_message, assistant_message in zip(user_messages, assistant_messages):
            processed_chat_history.append(
                f"{user_id} said to {assistant_id}, \"{user_message}\" and {assistant_id} replied, \"{assistant_message}\"."
            )
        processed_chat_history = "\n".join(processed_chat_history)

        summarize_prompt = PromptTemplate(SUMMARIZE_PROMPT_TEMPLATE).format(
            speaker=user_id,
            listener=assistant_id,
            input_sentence=memory_message,
            conversation_history=processed_chat_history,
        )
        transformed_memory_message = self.llm.complete(summarize_prompt)
        return transformed_memory_message.text

    async def async_transform_message_pair(
        self, message_pair: List[Dict[str, str]], user_id: str, assistant_id: str
    ):
        """Transform the user and assistant messages to a single memory message asynchronously."""
        return self.transform_message_pair(message_pair, user_id, assistant_id)


# See: https://www.mongodb.com/docs/languages/python/pymongo-driver/current/
class PersistentChatStore:
    """A class to store chat messages in a persistent database."""

    def __init__(self, uri=None, db_name=None, collection_name=None, use_async=False):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name

        if not use_async:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
        else:
            self.aclient = AsyncMongoClient(self.uri)
            self.adb = self.aclient[self.db_name]
            self.acollection = self.adb[self.collection_name]

    def save_chat(self, user_id: str, user_message: str, assistant_response: str):
        """Save a chat message of a user with the assistant."""
        chat_entry = {
            "user_id": user_id,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "timestamp": datetime.now(timezone.utc),
        }
        self.collection.insert_one(chat_entry)

    async def async_save_chat(
        self, user_id: str, user_message: str, assistant_response: str
    ):
        """Save a chat message of a user with the assistant asynchronously."""
        chat_entry = {
            "user_id": user_id,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "timestamp": datetime.now(timezone.utc),
        }
        await self.acollection.insert_one(chat_entry)

    def chat_messages_to_dict(self, messages):
        """Convert chat messages to LLM chat format."""
        messages_dict = []
        for message in messages:
            messages_dict.append(
                {"role": "assistant", "content": message["assistant_response"]}
            )
            messages_dict.append({"role": "user", "content": message["user_message"]})
        return messages_dict

    async def async_chat_messages_to_dict(self, messages):
        """Convert chat messages to LLM chat format with translation asynchronously."""
        messages_dict = []
        for message in messages:
            messages_dict.append({"role": "user", "content": message["user_message"]})
            messages_dict.append(
                {"role": "assistant", "content": message["assistant_response"]}
            )
        return messages_dict

    def get_chat_history(self, user_id: str, top_k: int = None) -> List[Dict[str, str]]:
        """Get the chat history of a user with the assistant."""
        if top_k:
            raw_history = list(
                self.collection.find({"user_id": user_id})
                .sort("timestamp", 1)
                .limit(top_k)
            )
            return self.chat_messages_to_dict(raw_history)

        raw_history = list(
            self.collection.find({"user_id": user_id}).sort("timestamp", 1)
        )
        return self.chat_messages_to_dict(raw_history)

    async def async_get_chat_history(
        self, user_id: str, top_k: int = None
    ) -> List[Dict[str, str]]:
        """Get the chat history of a user with the assistant asynchronously."""
        if top_k:
            raw_history = (
                await self.acollection.find({"user_id": user_id})
                .sort("timestamp", 1)
                .limit(top_k)
                .to_list(length=top_k)
            )
            return await self.async_chat_messages_to_dict(raw_history)

        raw_history = (
            await self.acollection.find({"user_id": user_id})
            .sort("timestamp", 1)
            .to_list(length=top_k)
        )
        return await self.async_chat_messages_to_dict(raw_history)

    def delete_chat_history(self, user_id: str):
        """Delete the chat history of a user with the assistant."""
        return self.collection.delete_many({"user_id": user_id}).deleted_count

    async def async_delete_chat_history(self, user_id: str):
        """Delete the chat history of a user with the assistant asynchronously."""
        return (await self.acollection.delete_many({"user_id": user_id})).deleted_count

    def count_messages(self, user_id: str):
        """Count the number of chat messages of a user in the collection."""
        return self.collection.count_documents({"user_id": user_id})

    async def async_count_messages(self, user_id: str):
        """Count the number of chat messages of a user in the collection asynchronously."""
        return await self.acollection.count_documents({"user_id": user_id})
