import os
from dotenv import load_dotenv
from chatbot.config.utils import check_bool


class Settings:
    """Settings for the chatbot service."""
    def __init__(self):

        # Load environment variables
        env_file = os.getenv("ENVIRONMENT_FILE", "/environment/.env")
        status = load_dotenv(env_file)
        if not status:
            raise Exception(f"Could not load environment variables from {env_file}")

        os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

        # Model serving settings
        self.EMBEDDER_SERVING_URL = os.getenv("EMBEDDER_SERVING_URL", "http://localhost:8011")
        self.RERANKER_SERVING_URL = os.getenv("RERANKER_SERVING_URL", "http://localhost:8012")
        self.LLM_SERVING_URL = os.getenv("LLM_SERVING_URL", "http://localhost:8013")
        self.LOCAL_LLM_MODEL_ID = os.getenv("LOCAL_LLM_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.EMBEDDER_MODEL_ID = os.getenv("EMBEDDER_MODEL_ID", "BAAI/llm-embedder")
        self.MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 256))
        self.LONG_MAX_NEW_TOKENS = int(os.getenv("LONG_MAX_NEW_TOKENS", 6000))
        self.USE_OPENAI_API = check_bool(os.getenv("USE_OPENAI_API", True))
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
        self.OTHER_API_KEY = os.getenv("OTHER_API_KEY", "EMPTY")
        self.RESPONSE_MODEL_ID = os.getenv("RESPONSE_MODEL_ID", "gpt-4o-mini")
        self.OPENAI_LLM_ID = os.getenv("OPENAI_LLM_ID", "gpt-4o-mini")
        self.OTHER_LLM_ID = os.getenv("OTHER_LLM_ID", "meta-llama/Llama-3.3-70B-Instruct")
        self.OPENAI_EMBEDDING_ID = os.getenv("OPENAI_EMBEDDING_ID", "text-embedding-3-small")

        # Vector store settings
        self.MILVUS_URL = os.getenv("MILVUS_URL", "http://localhost:8003")
        self.MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
        self.MINIO_ACCESS_KEY_ID = os.getenv("MINIO_ACCESS_KEY_ID")
        self.MINIO_SECRET_ACCESS_KEY = os.getenv("MINIO_SECRET_ACCESS_KEY")
        self.MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")

        # Cache chat store settings
        self.CHAT_STORE_HOST = os.getenv("CHAT_STORE_HOST", "localhost")
        self.CHAT_STORE_PORT = int(os.getenv("CHAT_STORE_PORT", 6379))
        self.CHAT_STORE_DB = int(os.getenv("CHAT_STORE_DB", 0))
        self.CHAT_STORE_USERNAME = os.getenv("CHAT_STORE_USERNAME")
        self.CHAT_STORE_PASSWORD = os.getenv("CHAT_STORE_PASSWORD")
        self.CHAT_STORE_TTL = int(os.getenv("CHAT_STORE_TTL", 86400))
        self.CHAT_STORE_MAX_MESSAGES_PAIRS = int(os.getenv("CHAT_STORE_MAX_MESSAGES_PAIRS", 5))

        # Persistent chat store settings
        self.PERSISTENT_CHAT_STORE_HOST = os.getenv("PERSISTENT_CHAT_STORE_HOST", "localhost")
        self.PERSISTENT_CHAT_STORE_PORT = int(os.getenv("PERSISTENT_CHAT_STORE_PORT", 27017))
        self.PERSISTENT_CHAT_STORE_USERNAME = os.getenv("ME_CONFIG_MONGODB_ADMINUSERNAME")
        self.PERSISTENT_CHAT_STORE_PASSWORD = os.getenv("ME_CONFIG_MONGODB_ADMINPASSWORD")
        self.PERSISTENT_CHAT_STORE_DB = os.getenv("PERSISTENT_CHAT_STORE_DB", "chatbot")
        self.PERSISTENT_CHAT_STORE_COLLECTION = os.getenv("PERSISTENT_CHAT_STORE_COLLECTION", "chat_history")

        # Graph store settings
        self.GRAPH_STORE_HOST = os.getenv("GRAPH_STORE_HOST", "localhost")
        self.GRAPH_STORE_PORT = int(os.getenv("GRAPH_STORE_PORT", 6379))
        self.GRAPH_STORE_USERNAME = os.getenv("GRAPH_STORE_USERNAME")
        self.GRAPH_STORE_PASSWORD = os.getenv("GRAPH_STORE_PASSWORD")
        self.GRAPH_STORE_BUILD = check_bool(os.getenv("GRAPH_STORE_BUILD", True))
        self.GRAPH_STORE_FORCE_BUILD = check_bool(os.getenv("GRAPH_STORE_FORCE_BUILD", False))
        self.MAX_TRIPLETS_PER_CHUNK = int(os.getenv("MAX_TRIPLETS_PER_CHUNK", 10))
        self.MAX_CLUSTER_SIZE = int(os.getenv("MAX_CLUSTER_SIZE", 10))
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
        self.DESCRIPTION_SUMMARIZATION_MAX_RETRIES = int(os.getenv("DESCRIPTION_SUMMARIZATION_MAX_RETRIES", 3))
        self.ENTITY_EXTRACTION_MAX_RETRIES = int(os.getenv("ENTITY_EXTRACTION_MAX_RETRIES", 1))

        # Retriever settings
        self.TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 5))
        self.PATH_DEPTH_GRAPH_RETRIEVAL = int(os.getenv("PATH_DEPTH_GRAPH_RETRIEVAL", 1))

SETTINGS = Settings()
