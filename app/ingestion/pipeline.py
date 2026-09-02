from langchain_core.documents import Document
from app.ingestion.loader import PDFLoader
from app.ingestion.splitter import DocumentSplitter
from app.embeddings.embeddings import EmbeddingService
from app.vectorstore.qdrant import QdrantVectorStore
from app.utils.hash_utils import get_pdf_collection_name

class IngestionPipeline:
    
    def __init__(self,loader : PDFLoader,splitter : DocumentSplitter, embeddings : EmbeddingService,vectorstore :  QdrantVectorStore):  
        self.loader=loader
        self.splitter=splitter
        self.embeddings=embeddings
        self.vectorstore=vectorstore
        self.collection_name=None
        
    
    def process(self,filepath:str):
        #loader
        print("2.came here ")
        collection_name=get_pdf_collection_name(filepath)
        self.collection_name=collection_name
        print("hashed collection name: ",collection_name) 
        documents=self.loader.load(filepath)
        
        if self.vectorstore.collection_exists(collection_name):
            print(
                f"Collection '{collection_name}' already exists."
            )
            print("Skipping ingestion.")
            
            return {
            "collection_name": collection_name,
            "status": "already_exists"
              }
            
            
        #splitter
        print("3.came here ")
        chunks=self.splitter.split(documents)
        
        
        
        #Embeddings
        print("4.came here ")
        chunk_PageContent=[chunk.page_content for chunk in chunks]
        embeddings=self.embeddings.embed_doucments(chunk_PageContent)
        self.vectorstore.create_collection(collection_name=collection_name)
        #vector store
        self.vectorstore.add_documents(collection_name=collection_name,documents=chunks,embeddings=embeddings)
        
        #retreive
        
        
        
        return chunks,embeddings