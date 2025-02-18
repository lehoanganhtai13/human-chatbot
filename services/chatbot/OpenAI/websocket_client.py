import websocket
import json
import time
import logging
import threading
from queue import Queue, Empty

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OPENAI_WEBSOCKET_URI = "wss://api.openai.com/v1/realtime?model="

class OpenAIWebSocketClient:
    def __init__(
        self,
        api_key: str,
        uri: str,
        headers: dict,
        timeout: int = 300,
        reconnect_delay: int = 5,
        retries: int = 2,
        logging: bool = False,
        debug: bool = False
    ):
        self.api_key = api_key
        self.uri = uri
        self.headers = headers
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self.retries = retries
        self.ws = None
        self.retry_attempts = retries
        self.lock = threading.Lock()
        self.token_queue = Queue()
        self.response_complete = threading.Event()
        self.logging = logging
        self.debug = debug

    def on_message(self, ws, message):
        message = json.loads(message)
        if message.get("type") == "response.text.delta":
            token = message.get("delta")
            self.token_queue.put(token)
        elif message.get("type") == "response.text.done":
            self.token_queue.put("[END]")
            self.response_complete.set()

    def on_error(self, ws, error):
        if self.debug:
            logging.error(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        if self.logging:
            logging.info(f"Connection closed: {close_status_code} - {close_msg}")

    def on_open(self, ws):
        if self.logging:
            logging.info("Connection opened")

    def on_ping(self, ws, message):
        if self.debug:
            logging.info(f"Ping received: {message}")

    def on_pong(self, ws, message):
        if self.debug:
            logging.info(f"Pong received: {message}")

    def create_connection(self):
        self.ws = websocket.WebSocketApp(
            self.uri,
            header=self.headers,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
            on_ping=self.on_ping,
            on_pong=self.on_pong
        )
        self.ws.run_forever(ping_interval=30, ping_timeout=10)

    def connect(self):
        while self.retry_attempts > 0:
            try:
                self.create_connection()
            except Exception as e:
                logging.error(f"Connection error: {e}")
                self.retry_attempts -= 1
                if self.retry_attempts > 0:
                    logging.info(f"Retrying connection in {self.reconnect_delay} seconds...")
                    time.sleep(self.reconnect_delay)

        if self.retry_attempts == 0:
            logging.error("Exceeded maximum retry attempts. Exiting...")

    def send_message(self, messages):
        with self.lock:
            if self.ws and self.ws.sock and self.ws.sock.connected:
                self.response_complete.clear()
                for message in messages:
                    self.ws.send(json.dumps(message))
            else:
                logging.error("WebSocket is not connected. Cannot send message.")

    def token_generator(self):
        while True:
            try:
                token = self.token_queue.get()
                yield token
                if self.response_complete.is_set() and token == "[END]":
                    break
            except Empty:
                if not (self.ws and self.ws.sock and self.ws.sock.connected):
                    break

if __name__ == "__main__":
    OPENAI_API_KEY = "sk-"
    HEADERS = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }
    client = OpenAIWebSocketClient(
        api_key=OPENAI_API_KEY,
        uri=OPENAI_WEBSOCKET_URI,
        headers=HEADERS
    )

    # Start the connection in a separate thread
    threading.Thread(target=client.connect).start()

    # Example usage: send a message
    time.sleep(5)  # Wait for the connection to establish

    test_times = 3
    while True:
        messages = [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Remember that you are Jensen and you are an AI data scientist at NVIDIA Company!",
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
                            "text": "Who are you?",
                        }
                    ]
                }
            },
            {
                "type": "response.create",
                "response": {"modalities": ["text"]}
            }
        ]
        client.send_message(messages)

        print("====================================")
        print("Response: ", end="", flush=True)
        # Use the token generator to print tokens
        for token in client.token_generator():
            if token == "[END]":
                print("\n====================================\n")
                continue
            print(token, end="", flush=True)

        test_times -= 1
        if test_times <= 0:
            print("Finished testing. Closing connection...")
            client.ws.close()
            break
        
        print("Waiting for 90 seconds before next test...")
        for i in range(90):
            time.sleep(1)
            print(".", end="", flush=True)
        print("\n\n")
