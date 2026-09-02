from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentSplitter:
    def __init__(self,chunk_size: int=1000,overlap:int=200):
        self.splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        
        
    def split(self,documents:list[Document])->list[Document]:
        return self.splitter.split_documents(documents)
        