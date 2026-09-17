from langchain_core.documents import Document
from app.ingestion.loader import PDFLoader
from app.ingestion.splitter import DocumentSplitter
from app.embeddings.embeddings import EmbeddingService
from app.vectorstore.qdrant import QdrantVectorStore
from app.utils.hash_utils import get_pdf_collection_name
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.bm25_store import BM25Store

class IngestionPipeline:
    
    def __init__(
        self,
        loader: PDFLoader,
        splitter: DocumentSplitter,
        embeddings: EmbeddingService,
        vectorstore: QdrantVectorStore,
        bm25retriever: BM25Retriever | None = None,
        bm25store: BM25Store | None = None,
    ):  
        self.loader = loader
        self.splitter = splitter
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25retriever
        self.bm25_store = bm25store or (bm25retriever.bm25_store if bm25retriever and hasattr(bm25retriever, "bm25_store") else None)
        self.collection_name = None

        
    
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
            # -
            return {
            "collection_name": collection_name,
            "status": "already_exists"
              }
            
            
        #splitter
        print("3.came here ")
        chunks = self.splitter.split(documents)
        
        if self.bm25_store:
            self.bm25_store.create_index(collection_name, chunks)
        # Embeddings
        print("4.came here ")
        chunk_PageContent = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_doucments(chunk_PageContent)
        self.vectorstore.create_collection(collection_name=collection_name)
        # vector store
        self.vectorstore.add_documents(collection_name=collection_name, documents=chunks, embeddings=embeddings)
        
        return chunks, embeddings

    def process_bytes(self, data: bytes, filename: str | None = None):
        """Process PDF given as bytes. Returns same as process()."""
        # compute collection name from filename + hash of bytes
        import hashlib, re, os

        base_name = os.path.splitext(os.path.basename(filename or "uploaded"))[0]
        hasher = hashlib.sha256()
        hasher.update(data)
        file_hash = hasher.hexdigest()[:12]
        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", base_name).lower()
        collection_name = f"doc_{clean_name[:20]}_{file_hash}"

        self.collection_name = collection_name

        documents = self.loader.load_bytes(data, source_name=filename)

        if self.vectorstore.collection_exists(collection_name):
            return {
                "collection_name": collection_name,
                "status": "already_exists",
            }

        chunks = self.splitter.split(documents)
        if self.bm25_store:
            self.bm25_store.create_index(collection_name, chunks)

        chunk_PageContent = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_doucments(chunk_PageContent)
        self.vectorstore.create_collection(collection_name=collection_name)
        self.vectorstore.add_documents(collection_name=collection_name, documents=chunks, embeddings=embeddings)

        return chunks, embeddings