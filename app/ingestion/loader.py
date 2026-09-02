from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFLoader:
    
    def load(self,file_path: str):
        path=Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        
        if path.suffix.lower()!=".pdf":
            raise ValueError("Only pdf files are supported")
        
        loader=PyPDFLoader(str(path))
        
        documents=loader.load()
        #This gives me the list of documents
        return documents
    
"""    
[
    Document(
        page_content="This is page 1...",
        metadata={
            "source": "document.pdf",
            "page": 0
        }
    ),

    Document(
        page_content="This is page 2...",
        metadata={
            "source": "document.pdf",
            "page": 1
        })
] """ 