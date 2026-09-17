import pytest
from pathlib import Path
from app.embeddings.embeddings import EmbeddingService
from app.vectorstore.qdrant import QdrantVectorStore
from app.retrieval.retriever import VectorRetriever
from app.generation.llm_service import LLMservice
from app.rag.RAGPipeline import RAGPipeline
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.bm25_store import BM25Store
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.loader import PDFLoader
from app.ingestion.splitter import DocumentSplitter


def test_verify_rag_pipeline():
    embedding_service = EmbeddingService()
    vector_store = QdrantVectorStore()
    bm25_store = BM25Store()
    bm25_retriever = BM25Retriever(bm25_store=bm25_store)
    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        embedding_model=embedding_service,
        score_threshold=None
    )
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever
    )
    llm_service = LLMservice()
    
    rag_pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        llm_service=llm_service
    )
    
    # Ingest a sample PDF if needed
    sample_pdf = Path("data/documents/sample_large.pdf")
    if not sample_pdf.exists():
        sample_pdf = next(Path("data/documents").glob("*.pdf"))

    with open(sample_pdf, "rb") as f:
        data = f.read()

    pipeline = IngestionPipeline(
        loader=PDFLoader(),
        splitter=DocumentSplitter(),
        embeddings=embedding_service,
        vectorstore=vector_store,
        bm25retriever=bm25_retriever,
        bm25store=bm25_store,
    )
    pipeline.process_bytes(data, filename=sample_pdf.name)
    collection_name = pipeline.collection_name

    # Ask question
    response = rag_pipeline.ask(
        collection_name=collection_name,
        query="What is this document about?",
        top_k=4
    )
    
    assert response is not None
    assert "answer" in response or "Answer" in response
    print("\nANSWER:", response.get("answer") or response.get("Answer"))
    print("\nSOURCES COUNT:", len(response.get("sources") or response.get("Sources") or []))


if __name__ == "__main__":
    test_verify_rag_pipeline()