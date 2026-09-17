from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings


class LLMservice:
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL_NAME,
            api_key=settings.GEMINI_API_KEY,
        )
        
    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        if isinstance(response.content, list):
            parts = []
            for item in response.content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(response.content)