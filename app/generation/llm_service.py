from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings




class LLMservice:
    
    def __init__(self):
        self.llm=ChatGoogleGenerativeAI(model=settings.GEMINI_MODEL_NAME)
        
        
        
    def generate(self,prompt:str)->str:
        response=self.llm.invoke(prompt)
        return response.content
    
    
    