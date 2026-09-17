from pathlib import Path
from app.ingestion.splitter import DocumentSplitter 
from app.ingestion.loader import PDFLoader
from app.embeddings.embeddings import EmbeddingService


def test_loader_and_splitter():
    loader = PDFLoader()
    splitter = DocumentSplitter(chunk_size=500, overlap=50)

    pdf_path = Path("data/documents/sample_large.pdf")
    if not pdf_path.exists():
        pdf_path = next(Path("data/documents").glob("*.pdf"))

    documents = loader.load(str(pdf_path))
    assert len(documents) > 0

    chunks = splitter.split(documents)
    assert len(chunks) > 0
    print(f"Loaded {len(documents)} pages, split into {len(chunks)} chunks.")


def test_embedding_service():
    embedder = EmbeddingService()
    query_vec = embedder.embed_query("Hello world")
    assert query_vec is not None
    assert len(query_vec) == 384


if __name__ == "__main__":
    test_loader_and_splitter()
    test_embedding_service()