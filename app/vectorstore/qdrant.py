import os
from typing import Any, List
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from app.core.config import settings

class QdrantVectorStore:

    def __init__(
        self,
        vector_size: int = 384,
        url: str | None = settings.QDRANT_URL,
        api_key: str | None = settings.QDRANT_API_KEY,
        local_path: str = "data/qdrant_db",
    ):
        self.vector_size = vector_size
        self.url = url
        self.api_key = api_key
        self.local_path = local_path
        self.client = None

        if url and api_key:
            print("Connecting to Qdrant Cloud...")
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            print(f"Connecting to Local Qdrant at {local_path}...")
            os.makedirs(local_path, exist_ok=True)
            self.client = QdrantClient(path=local_path)

    

    def create_collection(self,collection_name:str) -> None:
        try:
            #an additionla check here but its already being checked in the ingestion pipline 
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    ),
                )
                print(f"Created new collection: {collection_name}")
            else:
                print(f"Loaded existing collection: {collection_name}")
        except Exception as e:
            print(f"Error initializing Qdrant vector store: {e}")
            raise

    def add_documents(
        self,collection_name:str, documents: List[Any], embeddings: np.ndarray | list
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Count of texts and embeddings must match.")
        
        print(
            f"Adding {len(documents)} documents to Qdrant vector store '{collection_name}'..."
        )

        embeddings_list = (
            embeddings.tolist()
            if isinstance(embeddings, np.ndarray)
            else embeddings
        )
        points = []

        for i, (doc, embeddings_vec) in enumerate(
            zip(documents, embeddings_list)
        ):
            doc_id = str(uuid.uuid4())
            text_content = getattr(doc, "page_content", str(doc))
            metadata = dict(getattr(doc, "metadata", {}))
            metadata["doc_index"] = i
            metadata["content_length"] = len(text_content)

            payload = {"text": text_content, **metadata}

            point = PointStruct(
                id=doc_id, vector=embeddings_vec, payload=payload
            )
            points.append(point)

        try:
            self.client.upsert(
                collection_name=collection_name, points=points,timeout=60
            )
            print(
                f"Successfully added {len(points)} documents to Qdrant vector store"
            )
            count_info = self.client.count(collection_name=collection_name)
            print(f"Total documents in collection: {count_info.count}")
        except Exception as e:
            print(f"Error while adding the documents: {e}")
            raise

    def similarity_search(
        self,collection_name:str ,query_embeddings: list[float] | np.ndarray, top_k: int = 5
    ) -> list[dict]:
        query_vector = (
            query_embeddings.tolist()
            if isinstance(query_embeddings, np.ndarray)
            else query_embeddings
        )

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
        )

        results = []
        for hit in response.points:
            results.append(
                {
                    "id": hit.id,
                    "text": hit.payload.get("text", ""),
                    "score":hit.score,
                    "metadata": {
                        k: v for k, v in hit.payload.items() if k != "text"
                    },
                }
            )

        return results
    
    def collection_exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(
        collection_name=collection_name
    )
    
    # def get_collection_name()
    
        
            
            
        
        
        
#payload = {"text": text_content, **metadata}
# text_content = "Python is a programming language."
#metadata = {
#"page": 1,
# "doc_index": 0,
# "content_length": 33
# }
#         payload = {
#     "text": "Python is a programming language.",
#     "page": 1,
#     "doc_index": 0,
#     "content_length": 33
# }
# ┌─────────────────────────────────────────────┐
# │                  POINT                      │
# ├─────────────────────────────────────────────┤
# │ id      → "550e8400-e29b-..."               │
# │                                             │
# │ vector  → [0.12, 0.45, 0.78, ...]           │
# │                                             │
# │ payload → {                                  │
# │             "text": "Python is...",         │
# │             "page": 1,                      │
# │             "doc_index": 0,                 │
# │             "content_length": 33             │
# │           }                                  │
# └─────────────────────────────────────────────┘



# After all the loop stuff the points looks like this 

# points = [
#     PointStruct(
#         id="abc",
#         vector=[0.12, 0.45, 0.78, ...],
#         payload={
#             "text": "Python is a programming language.",
#             "page": 1,
#             "doc_index": 0,
#             "content_length": 33
#         }
#     ),

#     PointStruct(
#         id="def",
#         vector=[0.32, 0.11, 0.91, ...],
#         payload={
#             "text": "Python supports object oriented programming.",
#             "page": 2,
#             "doc_index": 1,
#             "content_length": 45
#         }
#     ),

#     PointStruct(
#         id="ghi",
#         vector=[0.72, 0.44, 0.15, ...],
#         payload={
#             "text": "Python has lists, dictionaries and tuples.",
#             "page": 3,
#             "doc_index": 2,
#             "content_length": 43
#         }
#     )
# ]