import os
import io
import time
from datetime import timedelta

from dotenv import load_dotenv

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from deep_translator import GoogleTranslator


from config import settings
from wrappers import Generator, LLMCore, ChatStore

@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator
    global chat_store
    global en_translator
    global ko_translator

    os.environ["NO_PROXY"] = "172.16.87.91"

    print("Initializing the translator...")
    en_translator = GoogleTranslator(source="auto", target="en")
    ko_translator = GoogleTranslator(source="auto", target="ko")

    print("Initializing the chat store...")
    chat_store = ChatStore(
        host=settings.CHAT_STORE_HOST,
        port=int(settings.CHAT_STORE_PORT),
        db=int(settings.CHAT_STORE_DB),
        username=settings.CHAT_STORE_USERNAME,
        password=settings.CHAT_STORE_PASSWORD,
        live_time_seconds=int(settings.CHAT_STORE_TTL),
        max_messages_pairs=int(settings.CHAT_STORE_MAX_MESSAGES_PAIRS)
    )

    print("Initializing the generator...")
    llm = LLMCore(
        uri=settings.LLM_SERVING_URL,
        model_id=settings.LLM_MODEL_ID,
        max_new_tokens=settings.LLM_MAX_LENGTH
    )
    generator = Generator(llm, chat_store=chat_store, max_new_tokens=settings.LLM_MAX_LENGTH, streaming=True)

    # Clean up the resources
    yield
    del generator
    del chat_store

app = FastAPI(lifespan=lifespan)

def capitalize_sentences(text):
    sentences = sent_tokenize(text)
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
    return ' '.join(capitalized_sentences)

@app.post("/query")
async def generate(user_id: str, query: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID cannot be empty")

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Detect the language of the query
    query = capitalize_sentences(query)
    
    lan = "en"
    en_text = en_translator.translate(query)
    if en_text != query:
        ko_text = ko_translator.translate(query)
        if ko_text != query:
            return {"language": "other", "text": "Sorry dad, I can only understand English and Korean."}
        else:
            lan = "ko"

            # Mix the user ID with the query
            mixed_query = f"|{user_id}|{query}"

            generator.streaming = False
            text = generator.generate(query=mixed_query, nodes=[NodeWithScore(node=Node(text=" "), score=0)])
            generator.streaming = True

            # Translate the response back to the user's language
            text = capitalize_sentences(ko_translator.translate(text))

            return {"language": lan, "text": text}

    # Mix the user ID with the query
    mixed_query = f"|{user_id}|{query}"

    # Generate a response stream
    streamer = generator.generate(query=mixed_query, nodes=[NodeWithScore(node=Node(text=" "), score=0)])
    return StreamingResponse(streamer)

@app.post("/add_message_pair")
async def add_message_pair(user_id: str, query: str, response: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID cannot be empty")

    if not query or not response:
        raise HTTPException(status_code=400, detail="Query and response cannot be empty")

    # Add the message pair to the chat store
    return await chat_store.async_add_message_pair(user_id, query, response)

@app.get("/get_chat_history")
async def get_chat_history(user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID cannot be empty")

    # Get the chat history
    return await chat_store.async_get_chat_history(user_id)


if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("app:app", host="0.0.0.0", port=8003)

