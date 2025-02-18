from typing import List, Dict

from pymongo import AsyncMongoClient, MongoClient
from datetime import datetime, timezone


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
