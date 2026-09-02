import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    #Qdrant Settings
    QDRANT_URL : str | None = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    QDRANT_LOCAL_PATH: str =os.getenv("QDRANT_LOCAL_PATH","data/qdrant_db")
    QDRANT_COLLECTION:str= os.getenv("QDRANT_COLLECTION_NAME", "pdf_documents"
    )
    # Model Settings
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
    )
    EMBEDDING_VECTOR_SIZE: int = int(os.getenv("EMBEDDING_VECTOR_SIZE", "384"))

    # LLM Settings (Groq / OpenAI)
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    
    # LLM_PROVIDER:str=os.getenv("LLM_PROVIDER",'gemini')
    GEMINI_API_KEY:str |None =os.getenv("GEMINI_API_KEY")
    
    GEMINI_MODEL_NAME:str=os.getenv("GEMINI_MODEL_NAME","gemini-3.6-flash")
    
   
    
    
   


# Global settings instance
settings = Settings()