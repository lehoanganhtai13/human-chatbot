from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore") # Specify the path to the .env file

    # Model settings
    EMBEDDER_SERVING_URL: str = Field("http://localhost:8000", env="EMBEDDER_SERVING_URL")
    LLM_SERVING_URL: str = Field("http://localhost:8001", env="LLM_SERVING_URL")
    RERANKER_SERVING_URL: str = Field("http://localhost:8002", env="RERANKER_SERVING_URL")
    LLM_MODEL_ID: str = Field("meta-llama/Meta-Llama-3.1-8B-Instruct")
    LLM_MAX_LENGTH: int = Field(512)

    # Vector database and storage settings
    MILVUS_URL: str = Field("http://localhost:8003", env="MILVUS_URL")
    MINIO_URL: str = Field("http://localhost:9000", env="MINIO_URL")
    MINIO_ACCESS_KEY_ID: str = Field(env="MINIO_ACCESS_KEY_ID")
    MINIO_SECRET_ACCESS_KEY: str = Field(env="MINIO_SECRET_ACCESS_KEY")
    MINIO_BUCKET_NAME: str = Field(env="MINIO_BUCKET_NAME")

    # Chat store settings
    CHAT_STORE_HOST: str = Field("localhost", env="CHAT_STORE_HOST")
    CHAT_STORE_PORT: int = Field(6379, env="CHAT_STORE_PORT")
    CHAT_STORE_DB: int = Field(0, env="CHAT_STORE_DB")
    CHAT_STORE_USERNAME: str = Field(env="CHAT_STORE_USERNAME")
    CHAT_STORE_PASSWORD: str = Field(env="CHAT_STORE_PASSWORD")
    CHAT_STORE_TTL: int = Field(86400, env="CHAT_STORE_TTL")
    CHAT_STORE_MAX_MESSAGES_PAIRS: int = Field(5, env="CHAT_STORE_MAX_MESSAGES_PAIRS")

settings = Settings()
