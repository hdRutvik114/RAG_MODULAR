from app.embeddings.embeddings import EmbeddingService
from app.vectorstore.qdrant import QdrantVectorStore
from app.retrieval.retriever import VectorRetriever
from app.generation.llm_service import LLMservice
from app.rag.RAGPipeline import RAGPipeline
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.bm25_store import BM25Store
# Dependencies
def test_verify_rag_pipeline():
    
    embedding_service = EmbeddingService()
    
    vector_store = QdrantVectorStore()
    
    
    retriever = VectorRetriever(
        vector_store=vector_store,
        embedding_model=embedding_service,
        score_threshold=None
    )
    # key word based Finding inside the corpus or vector database
    bm25_store=BM25Store()
    llm_service = LLMservice()
    
    bm25retriever=BM25Retriever(
        bm25_store=bm25_store
    )
    
    
    hybrid_retriever = HybridRetriever(
    vector_retriever=retriever,
    bm25_retriever=bm25retriever
)
    # Compose the application
    rag_pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        llm_service=llm_service
    )
    
    ddd=0
    # Ask question
    response = rag_pipeline.ask(
        collection_name="doc_ignitedminds_fb4b60aecb78",
        query="How does self-attention work?",
        top_k=4
    )
    
    print("\nANSWER:")
    
    # rutivk
    print(response["Answer"])
    
    print("\nSOURCES:")
    for source in response["Sources"]:
        print(source["score"], source["text"][:100])
        
    


if __name__ == "__main__":
    # Call your functions here to execute them directly
    test_verify_rag_pipeline()  # Replace with your actual function name inside test_rag.py