from app.embeddings.embeddings import EmbeddingService
from app.vectorstore.qdrant import QdrantVectorStore
from app.retrieval.retriever import VectorRetriever
from app.generation.llm_service import LLMservice
from app.rag.RAGPipeline import RAGPipeline
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

# Dependencies
def fun():
    
    embedding_service = EmbeddingService()
    
    vector_store = QdrantVectorStore()
    
    
    retriever = VectorRetriever(
        vector_store=vector_store,
        embedding_model=embedding_service,
        score_threshold=None
    )
    bm25_retriever=BM25Retriever()
    llm_service = LLMservice()
    
    hybrid_retriever = HybridRetriever(
    vector_retriever=retriever,
    bm25_retriever=
)
    # Compose the application
    rag_pipeline = RAGPipeline(
        retriever=hybrid_retiever,
        llm_service=llm_service
    )
    
    
    # Ask question
    response = rag_pipeline.ask(
        collection_name="doc_attention_paper_bdfaa68d8984",
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
    fun()  # Replace with your actual function name inside test_rag.py