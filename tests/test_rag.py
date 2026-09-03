from app.embeddings.embeddings import EmbeddingService
from app.vectorstore.qdrant import QdrantVectorStore
from app.retrieval.retriever import VectorRetriever
from app.generation.llm_service import LLMService
from app.rag.RAGPipeline import RAGPipeline


# Dependencies
embedding_service = EmbeddingService()

vector_store = QdrantVectorStore()

retriever = VectorRetriever(
    vector_store=vector_store,
    embedding_model=embedding_service,
    score_threshold=0.6
)

llm_service = LLMService()


# Compose the application
rag_pipeline = RAGPipeline(
    retriever=retriever,
    llm_service=llm_service
)


# Ask question
response = rag_pipeline.ask(
    collection_name="doc_attention_paper_bdfaa68d8984",
    query="How does scaled dot product attention work?",
    top_k=4
)

print("\nANSWER:")

# rutivk
print(response["answer"])

print("\nSOURCES:")
for source in response["sources"]:
    print(source["score"], source["text"][:100])