class CallServerLLMError(Exception):
    """Exception raised for errors in the call to the LLM server."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class FormatChatMessageError(Exception):
    """Exception raised for errors in formatting the chat message."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class CallServerEmbedderError(Exception):
    """Exception raised for errors in the call to the Embedder server."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InitWebsocketOpenAIError(Exception):
    """Exception raised for errors in initializing the websocket client for OpenAI."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
