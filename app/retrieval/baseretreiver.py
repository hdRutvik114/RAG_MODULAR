from abc import ABC,abstractmethod
from typing import List,Dict,Any





class BaseRetriever(ABC):
    
    @abstractmethod
    def query_retriever(self,query:str,top_k :int=4):
        pass
        

 