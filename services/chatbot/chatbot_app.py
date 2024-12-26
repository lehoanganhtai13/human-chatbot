import asyncio
from contextlib import asynccontextmanager
from typing import Dict
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from concurrent.futures import ThreadPoolExecutor
from asyncio import Lock, sleep
import time

from chatbot.server.chatbot_server import ChatbotServer
from chatbot.utils.graph_store import Neo4jGraphStore, FalkorDBGraphStore


class AsyncRWLock:
    """Asynchronous read-write lock."""
    def __init__(self):
        self._read_lock = Lock()
        self._write_lock = Lock()
        self._reader_count = 0

    @asynccontextmanager
    async def read_lock(self, timeout=5.0):
        try:
            start_time = time.time()
            await asyncio.wait_for(self._read_lock.acquire(), timeout)
            self._reader_count += 1
            if self._reader_count == 1:
                await self._write_lock.acquire()
            self._read_lock.release()
            yield
        except asyncio.TimeoutError:
            print(f"Read lock acquisition timed out after {time.time() - start_time:.2f}s")
            raise
        finally:
            await self._read_lock.acquire()
            self._reader_count -= 1
            if self._reader_count == 0:
                self._write_lock.release()
            self._read_lock.release()

    @asynccontextmanager
    async def write_lock(self, timeout=5.0):
        try:
            await asyncio.wait_for(self._write_lock.acquire(), timeout)
            yield
        finally:
            self._write_lock.release()

class ConnectionManager:
    """Manage WebSocket connections."""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.lock = AsyncRWLock()

    async def connect(self, key: str, websocket: WebSocket) -> WebSocket:
        await websocket.accept()
        async with self.lock.write_lock():
            old_websocket = self.active_connections.get(key)
            self.active_connections[key] = websocket
            return old_websocket

    async def disconnect(self, key: str):
        async with self.lock.write_lock():
            if key in self.active_connections:
                del self.active_connections[key]

    async def get_connection(self, key: str) -> WebSocket:
        async with self.lock.read_lock():
            return self.active_connections.get(key)

class ChatbotManager:
    def __init__(self):
        self.chatbots: Dict[str, ChatbotServer] = {}
        self.lock = AsyncRWLock()

    async def get_or_create(
        self, 
        key: str, 
        user_id: str, 
        avatar_id: str,
        use_default_story: bool = True,
        avatar_instruction_text: str = "",
    ) -> tuple[ChatbotServer, bool]:
        async with self.lock.write_lock():
            chatbot = self.chatbots.get(key)
            if not chatbot:
                print(f"Creating chatbot for {user_id} with avatar {avatar_id}")
                chatbot = ChatbotServer(
                    user_id=user_id,
                    avatar_name=avatar_id,
                    use_default_story=use_default_story,
                    avatar_instruction_text=avatar_instruction_text,
                    warm_up=False,
                )
                self.chatbots[key] = chatbot
                return chatbot, True
            return chatbot, False

    async def remove(self, key: str):
        async with self.lock.write_lock():
            if key in self.chatbots:
                chatbot = self.chatbots[key]
                chatbot.cleanup()
                del self.chatbots[key]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up Chatbot server...")
    global thread_pool
    global chatbot_manager
    global connection_manager

    # Global dictionaries to store WebSocket connections and chatbots
    chatbot_manager = ChatbotManager()
    connection_manager = ConnectionManager()

    # Shared thread pool for post-processing tasks
    thread_pool = ThreadPoolExecutor(max_workers=20)

    yield
    print("Shutting down Chatbot server...")
    thread_pool.shutdown(wait=True)

    del chatbot_manager
    del connection_manager
    del thread_pool


app = FastAPI(lifespan=lifespan)

@app.post("/offer/{user_id}/{avatar_id}")
async def offer(
    user_id: str,
    avatar_id: str,
    use_default_story: bool = True,
    avatar_instruction_text: str = "",
):
    """Offer to initialize chatbot for chat interaction."""
    chatbot_key = f"{user_id}_{avatar_id}"
    chatbot, is_new = await chatbot_manager.get_or_create(
        chatbot_key, user_id, avatar_id, use_default_story, avatar_instruction_text
    )
    return {"status": "success" if is_new else "exists"}

@app.websocket("/ws/chat/{user_id}/{avatar_id}")
async def websocket_chat(websocket: WebSocket, user_id: str, avatar_id: str):
    """Handle WebSocket connection for chatbot chat interaction."""
    connection_key = f"{user_id}_{avatar_id}"
    old_websocket = await connection_manager.connect(connection_key, websocket)

    try:
        # Close old connection if exists
        if old_websocket and old_websocket != websocket:
            print(f"Closing old websocket connection for {connection_key}")
            asyncio.create_task(old_websocket.close())

        # Get or create chatbot
        chatbot, _ = await chatbot_manager.get_or_create(connection_key, user_id, avatar_id)
        await websocket.send_text("ready")

        while True:
            if websocket.client_state == WebSocketState.DISCONNECTED:
                break

            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                continue

            if message == "ping":
                await websocket.send_text("pong")
                continue

            print(f"Message from {user_id}: {message}")
            
            try:
                streamer = await chatbot.chat(message)
                async for msg in streamer:
                    if websocket.client_state == WebSocketState.DISCONNECTED:
                        break
                    await websocket.send_text(msg)
                    await sleep(0.01)
                
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text("[END]")
                    thread_pool.submit(chatbot.post_processing)

            except Exception as e:
                print(f"Error processing message: {e}")
                await websocket.send_text(f"error: {str(e)}")

    except WebSocketDisconnect:
        print("===============================================")
        print(f"WebSocket disconnected for {connection_key}")
    except Exception as e:
        print("===============================================")
        print(f"Error in websocket handler: {e}")
    finally:
        await connection_manager.disconnect(connection_key)
        await chatbot_manager.remove(connection_key)
        print(f"Cleaned up resources for {connection_key}")
        print(f"Chatbot {avatar_id} of user {user_id} disconnected from chat.")
        print("===============================================")

@app.delete("/delete_chatbot/{user_id}/{avatar_id}")
async def delete_avatar(user_id: str, avatar_id: str):
    """Delete a chatbot for a user."""
    chatbot_key = f"{user_id}_{avatar_id}"
    
    try:
        # Delete the graph store if it is FalkorDB and clear the graph if it is Neo4j
        chatbot = ChatbotServer(user_id=user_id, avatar_name=avatar_id, warm_up=False)
        if isinstance(chatbot.graph_store, Neo4jGraphStore):
            print("Clearing Neo4j graph...")
            chatbot.graph_store.clear_graph()
        elif isinstance(chatbot.graph_store, FalkorDBGraphStore):
            print("Deleting FalkorDB graph...")
            chatbot.graph_store.delete_graph()

        # Clear the chat history and cache for the chatbot
        print(f"Clearing chat history and cache for {avatar_id} of user {user_id}...")
        chatbot.cache_chat_store.clear_messages(f"original_{user_id}_{avatar_id}")
        chatbot.cache_chat_store.clear_messages(f"translated_{user_id}_{avatar_id}")
        chatbot.persistent_chat_store.delete_chat_history(f"translated_{user_id}_{avatar_id}")

        # Remove from manager
        await chatbot_manager.remove(chatbot_key)
        
        return {"status": "success"}
    except Exception as e:
        print(f"Error deleting chatbot {chatbot_key}: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("chatbot/chatbot_app:app", host="0.0.0.0", port=8000)
