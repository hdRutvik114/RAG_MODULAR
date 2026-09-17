from app.retrieval.baseretreiver import BaseRetriever


class VectorRetriever(BaseRetriever):
    
    def __init__(self,vector_store,embedding_model,score_threshold :float |None =None):
        self.vectorstore=vector_store
        self.embedding_model=embedding_model
        self.score_threshold=score_threshold
        
    def query_retriever(self, collection_name: str, query: str, top_k: int = 4):
        query_embedding = self.embedding_model.embed_query(query)
        # Ensure query_embedding is 1D list/array
        if hasattr(query_embedding, "ndim") and query_embedding.ndim > 1:
            query_embedding = query_embedding[0]
        elif isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]
            
        results = self.vectorstore.similarity_search(
            collection_name=collection_name,
            query_embeddings=query_embedding,
            top_k=top_k
        )
        
        if self.score_threshold is not None:
            results = [
                result for result in results if result.get("score", 0) >= self.score_threshold
            ]
        
        return results