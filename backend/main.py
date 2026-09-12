from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.embeddings.embeddings import EmbeddingService
from app.generation.llm_service import LLMservice
from app.ingestion.loader import PDFLoader
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.splitter import DocumentSplitter
from app.rag.RAGPipeline import RAGPipeline
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.retriever import VectorRetriever
from app.utils.hash_utils import get_pdf_collection_name
from app.vectorstore.qdrant import QdrantVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_LOCAL_PATH = str(DATA_DIR / "qdrant_db")
BM25_STORE_PATH = str(DATA_DIR / "bm25")


class FallbackLLM:
    def generate(self, prompt: str) -> str:
        if "Context:" not in prompt:
            return "The document was indexed successfully, but no LLM provider is configured."

        context_part = prompt.split("Context:", 1)[1].split("Question:", 1)[0].strip()
        if not context_part:
            return "The document was indexed successfully, but the answer could not be generated without an LLM provider."
        return context_part[:400].strip() + "..."


class RAGBackendService:
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.vector_store = QdrantVectorStore(
            url=None,
            api_key=None,
            local_path=QDRANT_LOCAL_PATH,
        )
        self.bm25_store = BM25Store(base_path=BM25_STORE_PATH)
        self.bm25_retriever = BM25Retriever(bm25_store=self.bm25_store)
        self.vector_retriever = VectorRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_service,
            score_threshold=None,
        )
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
        )
        self.llm_service = self._build_llm_service()

    def _build_llm_service(self):
        try:
            return LLMservice()
        except Exception:
            return FallbackLLM()

    def ingest_pdf(self, file_path: Path) -> dict[str, Any]:
        pipeline = IngestionPipeline(
            loader=PDFLoader(),
            splitter=DocumentSplitter(),
            embeddings=self.embedding_service,
            vectorstore=self.vector_store,
            bm25retriever=self.bm25_retriever,
            bm25store=self.bm25_store,
        )
        pipeline.bm25_store = self.bm25_store

        result = pipeline.process(str(file_path))
        pdf_id = pipeline.collection_name or get_pdf_collection_name(str(file_path))

        if isinstance(result, dict):
            status = result.get("status", "indexed")
        else:
            status = "indexed"

        return {
            "pdf_id": pdf_id,
            "collection_name": pdf_id,
            "status": status,
            "message": "PDF uploaded and indexed successfully.",
        }

    def query_pdf(self, pdf_id: str, question: str, top_k: int = 4) -> dict[str, Any]:
        if not self.bm25_store.index_exists(pdf_id) and not self.vector_store.collection_exists(pdf_id):
            raise FileNotFoundError(f"Collection '{pdf_id}' was not found. Upload the PDF first.")

        rag_pipeline = RAGPipeline(
            retriever=self.hybrid_retriever,
            llm_service=self.llm_service,
        )

        response = rag_pipeline.ask(
            collection_name=pdf_id,
            query=question,
            top_k=top_k,
        )

        answer = response.get("Answer") or response.get("answer") or "I couldn't find a relevant answer in the document."
        if isinstance(answer, list):
            flat_parts = []
            for item in answer:
                if isinstance(item, dict):
                    flat_parts.append(str(item.get("text") or json.dumps(item, default=str)))
                else:
                    flat_parts.append(str(item))
            answer = " ".join(flat_parts)
        elif isinstance(answer, dict):
            answer = answer.get("text") or json.dumps(answer, default=str)

        sources = response.get("Sources", [])

        return {
            "pdf_id": pdf_id,
            "answer": str(answer),
            "sources": sources,
            "bm25_used": True,
        }


class QueryRequest(BaseModel):
    pdf_id: str
    question: str


backend_service = RAGBackendService()
app = FastAPI(title="RAG PDF API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "message": "RAG backend is running"}


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    safe_name = file.filename.replace(" ", "_")
    upload_path = UPLOAD_DIR / safe_name

    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = backend_service.ingest_pdf(upload_path)
        return result
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc


@app.post("/api/query")
async def query_pdf(payload: QueryRequest):
    if not payload.pdf_id or not payload.question:
        raise HTTPException(status_code=400, detail="pdf_id and question are required.")

    try:
        return backend_service.query_pdf(pdf_id=payload.pdf_id, question=payload.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
