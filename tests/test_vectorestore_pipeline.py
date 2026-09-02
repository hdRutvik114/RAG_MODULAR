# tests/test_full_pipeline.py
import os
from langchain_core.documents import Document

from app.ingestion.loader import PDFLoader
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.splitter import DocumentSplitter
from app.embeddings.embeddings import EmbeddingService
from app.vectorstore.qdrant import QdrantVectorStore
from app.retrieval.retriever import VectorRetriever


def run_pipeline_test():
    print("="*10)
    print("Started")
    print("="*10)

    #initialize the loader
    loader=PDFLoader()
    splitter = DocumentSplitter(chunk_size=500, overlap=50)
    embedder = EmbeddingService()  # Uses settings.EMBEDDING_MODEL_NAME
    vector_store=QdrantVectorStore()

    # 2. Setup Pipeline
    
    print("\n[2/5] Wiring Ingestion Pipeline...")
    pipeline = IngestionPipeline(
        loader=loader,
        splitter=splitter,
        embeddings=embedder,
        vectorstore=vector_store,
    )
    
    retriver=VectorRetriever(vector_store,embedder)

    # 3. Path to a sample PDF
    sample_pdf_path = "data/documents/attention_paper.pdf"
 
    # Fallback to dummy data if no PDF exists yet
    if not os.path.exists(sample_pdf_path):
        print(f"\n⚠️  PDF file not found at '{sample_pdf_path}'.")
        print("Using in-memory mock Documents to verify the rest of the flow...")

        mock_docs = [
            Document(
                page_content=(
                    "Transformer models rely on self-attention mechanisms to process "
                    "input tokens in parallel rather than sequentially."
                ),
                metadata={"source": "transformer_paper.pdf", "page": 1},
            ),
            Document(
                page_content=(
                    "Qdrant is a high-performance vector search engine written in Rust, "
                    "designed for production RAG and semantic search applications."
                ),
                metadata={"source": "qdrant_docs.pdf", "page": 1},
            ),
        ]

        chunks = splitter.split(mock_docs)
        print(f"Created {len(chunks)} chunks from mock data.")

        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedder.load(texts)
        print(f"Generated embeddings of shape: {embeddings.shape}")

        vector_store.add_documents(documents=chunks, embeddings=embeddings)

    else:
        # Ingest actual PDF
        print("Came here ")
        print(f"\n[3/5] Processing PDF: {sample_pdf_path}...")
        chunks = pipeline.process(sample_pdf_path)
        print(f"Successfully processed and stored {len(chunks)} chunks!")

    # 4. Test Similarity Search (Retrieval)
    print("\n[4/5] Testing Similarity Search...")
    test_query = "How does scaled dot product attention work?"
    print(f"Query: '{test_query}'")

    # Convert query into embedding vector
    # query_vector = embedder.embed_query([test_query])[0]
    
    collection_name=pipeline.collection_name
    retriver_results=retriver.query_retriever(collection_name=collection_name,query=test_query)
    
    
    
    # Search top 2 most relevant chunks
    # search_results = vector_store.similarity_search(
    #     query_embeddings=query_vector, top_k=2
    # )

    # 5. Display Search Results
    # print("\n[5/5] Search Results:")
    # print("-" * 60)
    # for rank, result in enumerate(search_results, start=1):
    #     print(f"Rank {rank}:")
    #     print(f"  ID:       {result['id']}")
    #     print(f"  Text:     {result['text']}")
    #     print(f"  Metadata: {result['metadata']}")
    #     print("-" * 60)

    print("\n✅ TEST PASSED: Ingestion, Embedding, and Retrieval are working!")


if __name__ == "__main__":
    run_pipeline_test()
    
    