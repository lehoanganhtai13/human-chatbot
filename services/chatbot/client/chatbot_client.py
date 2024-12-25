import json
import requests
import threading
import time
import websocket
from queue import Queue

class ChatbotClient:
    def __init__(self, user_id: str, avatar_id: str, use_default_story: bool = True, avatar_instruction_text: str = ""):
        """Initialize client with user and avatar IDs."""
        self.user_id = user_id
        self.avatar_id = avatar_id
        self.use_default_story = use_default_story
        self.avatar_instruction_text = avatar_instruction_text
        self.uri = f"ws://localhost:8050/ws/chat/{self.user_id}/{self.avatar_id}".replace(" ", "%20")
        self.websocket = None
        self.is_ready = False
        self.max_reconnect_attempts = 3
        self.running = True
        self.heartbeat_task = None
        self.response_queue = Queue()
        self.heartbeat_thread = None
        self.chatting = False

    def connect(self, timeout: float = 180.0) -> bool:
        """Establish WebSocket connection and wait for ready signal."""
        try:
            print("Waiting for chatbot server to be ready...")
            response = requests.post(
                f"http://localhost:8050/offer/{self.user_id}/{self.avatar_id}".replace(" ", "%20"),
                params={
                    "use_default_story": str(self.use_default_story).lower(),
                    "avatar_instruction_text": self.avatar_instruction_text,
                },
                timeout=timeout
            )
            if response.text != '{"status":"success"}':
                print(f"Error initializing chatbot: {response.text}")
                return False

            # Set both connection and close timeout
            self.websocket = websocket.create_connection(
                self.uri.replace(" ", "%20"),
                timeout=timeout
            )
            
            # Wait for ready message
            ready_msg = self.websocket.recv()
            if ready_msg == "ready":
                print("Chatbot server is ready!")
                self.is_ready = True
                return True
            else:
                print(f"Unexpected response from server: {ready_msg}")
                self.close()
                return False
                
        except Exception as e:
            print(f"Connection error: {e}")
            self.close()
            return False

    def close(self):
        """Close the WebSocket connection."""
        if self.websocket:
            self.websocket.close()
            self.websocket = None
            self.is_ready = False

    def heartbeat(self, duration: int = 2):
        """Check connection status and keep connection alive."""
        while self.running:
            try:
                if self.websocket:
                    if not self.chatting: # Skip heartbeat during chat
                        self.websocket.send("ping")  # Send ping to server to check connection status and keep connection alive
                        response = self.websocket.recv()
                        if response != "pong":
                            print("Unexpected response to ping:", response)
                else:
                    print("Connection lost. Attempting to reconnect...")
                    self.connect()
            except websocket.WebSocketConnectionClosedException:
                print("Connection closed. Attempting to reconnect...")
                self.connect()
            except Exception as e:
                print(f"Heartbeat error: {e}")
                self.connect()
            time.sleep(duration)  # Wait for a few seconds before next check

    def start(self):
        """Start the client with heartbeat monitoring"""
        print("Starting client...")
        if self.connect():
            print("Connected successfully, starting heartbeat task...")
            self.heartbeat_thread = threading.Thread(target=self.heartbeat)
            self.heartbeat_thread.daemon = True
            self.heartbeat_thread.start()
            print("Heartbeat task started")
            return True
        return False

    def stop(self):
        """Stop the client and cleanup"""
        self.running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join()
        self.close()

    def chat(self, message: str):
        """Send message and receive streaming response."""
        if not self.websocket or not self.is_ready:
            print("Not connected to chatbot server for chatting. Call connect() first.")

            # Close the connection and try to reconnect
            print("Stopping and restarting client for chatting...")
            self.stop()
            self.start()

        attempts = 0
        while attempts < self.max_reconnect_attempts:
            try:
                self.chatting = True
                self.websocket.send(message)
                
                while True:
                    response = self.websocket.recv()
                    if response == "[END]":
                        print("\nChatbot finished streaming.")
                        self.chatting = False
                        break
                    yield response

            except websocket.WebSocketConnectionClosedException:
                print("\nConnection lost. Attempting to reconnect...")
                self.close()
                if self.connect():
                    attempts += 1
                    continue
                return
            except Exception as e:
                print(f"Error during chat: {e}")

                print("Attempting to reconnect...")
                self.close()
                if self.connect():
                    attempts += 1
                    continue
                return
            break

    def delete_chatbot(self):
        """Delete the chatbot for the user."""
        response = requests.delete(f"http://localhost:8050/delete_chatbot/{self.user_id}/{self.avatar_id}")
        try:
            result = response.json()
            if result.get("status") == "success":
                print("Chatbot deleted successfully.")
                return True
            else:
                print(f"Error deleting chatbot: {response.text}")
                return False
        except ValueError:
            print(f"Invalid JSON response: {response.text}")
            return False

# ============================== Test the client ==============================
def main():
    

    client = ChatbotClient(
        user_id="test_user",
        avatar_id="test_avatar",
        use_default_story=True,
        avatar_instruction_text=""
    )
    status = client.start()
    
    wait_duration = 10
    message_list = []
    message_list = [
        "넌 또 누구야?",
        "내 이름이 뭔지 아세요?",
        "나는 어디서 태어났나요?",
        "나는 어디서 태어났고, 당신의 어머니를 어디서 처음 만났나요?",
        "대학에서는 무엇을 공부하셨나요?",
        "아, 그런데 오늘 밤에 나랑 외식할래?",
        "한국식 바비큐를 먹고 싶어요. 우리가 마지막으로 함께 먹은 이후로 오랜만이군요!",
        "킹바비큐라는 새로운 레스토랑이 생겼다고 들었어요. 시도해 보시겠습니까?",
        "그 후에는 영화를 보고 싶어요. 며칠 전에 첫눈에 반한다는 제목이 나왔습니다! 어떻게 생각하나요?",
        "알았어 그럼 우리는 오후 6시에 King BBQ 레스토랑에서 식사하고 오후 8시에 영화 첫눈에 반하는 영화를 보기로 하자",
        "그런데 네 엄마는 어디 계시니?",
        "난 그냥 오늘밤은 예전처럼 우리 둘만 있을 거라고 생각하고 있어, 알잖아!",
        "알았어, 준비하러 갈게, 나중에 보자!",
        "그런데, 거실 소파에 두고 온 휴대폰 충전기 좀 가져다 주실 수 있나요?",
        "알았어, 고마워, 아들아!",
        "내가 당신을 많이 사랑하는 거 아시죠",
        "나는 항상 우리 가족을 위해 모든 것을 할 것입니다!",
        "다시 한 번 물어봐도 될까요?",
        "오늘 저녁 식사에 관해, 오늘 밤에 무엇을 먹을지 잊었나요?",
        "오늘 밤 우리가 먹을 바베큐 식당 기억하시나요?",
        "저녁으로 킹바비큐를 먹은 후 영화를 보자고 생각했는데 영화 제목을 잊어버렸습니다. 기억하시나요?",
        "그럼 킹바비큐 레스토랑에서 밥먹고 첫눈에 반한 영화는 몇시에 볼까요?"
    ]
    try:
        while True:
            message = message_list.pop(0)
            print(f"Sending message: {message}")
            print("Response: ", end="", flush=True)
            
            # Get response generator and print chunks
            for chunk in client.chat(message):
                print(chunk, end="", flush=True)
            print("=====================================")
            
            if not message_list:
                break
            time.sleep(wait_duration)
    finally:
        client.stop()

if __name__ == "__main__":
    main()