from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List
from pypdf import PdfReader
from io import BytesIO


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

    def load_bytes(self, data: bytes, source_name: str | None = None) -> List[Document]:
        """Load PDF from bytes and return a list of langchain_core.documents.Document."""
        reader = PdfReader(BytesIO(data))
        documents: List[Document] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            metadata = {"source": source_name or "pdf_bytes", "page": i}
            documents.append(Document(page_content=text, metadata=metadata))

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