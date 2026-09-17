import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from workspace root and app/.env
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings:
    # Qdrant Settings
    QDRANT_URL: str | None = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    QDRANT_LOCAL_PATH: str = os.getenv("QDRANT_LOCAL_PATH", "data/qdrant_db")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION_NAME", "pdf_documents")

    # Model Settings
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
    )
    EMBEDDING_VECTOR_SIZE: int = int(os.getenv("EMBEDDING_VECTOR_SIZE", "384"))

    # LLM Settings (Groq / OpenAI / Gemini)
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-3.6-flash")


# Global settings instance
settings = Settings()

# Ensure GOOGLE_API_KEY is available in os.environ for Google GenAI SDKs if set
if settings.GEMINI_API_KEY and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY