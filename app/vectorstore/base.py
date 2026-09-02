from abc import ABC,abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self,collection_name:str,text:list[str],embeddings:Any)->None:
        pass
    
    
    @abstractmethod
    def similarity_search(self,query_embeddings:list[float],top_k:int=5)->list[dict]:
        pass 
        