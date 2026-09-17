from pathlib import Path
from langchain_core.documents import Document

from app.ingestion.loader import PDFLoader
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.splitter import DocumentSplitter
from app.embeddings.embeddings import EmbeddingService
from app.vectorstore.qdrant import QdrantVectorStore
from app.retrieval.retriever import VectorRetriever


def test_vectorestore_pipeline():
    loader = PDFLoader()
    splitter = DocumentSplitter(chunk_size=500, overlap=50)
    embedder = EmbeddingService()
    vector_store = QdrantVectorStore()

    pipeline = IngestionPipeline(
        loader=loader,
        splitter=splitter,
        embeddings=embedder,
        vectorstore=vector_store,
    )
    
    retriever = VectorRetriever(vector_store, embedder)

    sample_pdf = Path("data/documents/sample_large.pdf")
    if not sample_pdf.exists():
        sample_pdf = next(Path("data/documents").glob("*.pdf"))

    with open(sample_pdf, "rb") as f:
        data = f.read()

    pipeline.process_bytes(data, filename=sample_pdf.name)
    collection_name = pipeline.collection_name

    test_query = "What is in this document?"
    results = retriever.query_retriever(collection_name=collection_name, query=test_query, top_k=2)

    assert results is not None
    assert len(results) > 0
    print("\n✅ Ingestion, Embedding, and Retrieval are working successfully!")


if __name__ == "__main__":
    test_vectorestore_pipeline()