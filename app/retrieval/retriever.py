from app.retrieval.baseretreiver import BaseRetriever


class VectorRetriever(BaseRetriever):
    
    def __init__(self,vector_store,embedding_model,score_threshold :float |None =None):
        self.vectorstore=vector_store
        self.embedding_model=embedding_model
        self.score_threshold=score_threshold
        
    def query_retriever(self,collection_name:str,query:str,top_k:int=4):
        query_embedding=self.embedding_model.embed_query([query])[0]
        results=self.vectorstore.similarity_search(collection_name=collection_name,query_embeddings=query_embedding,top_k=top_k)
        
        
        if self.score_threshold is not None:
            results=[result 
                for result in results if result['score']>=self.score_threshold
                
            ]
        if not results:
            return []
        
        for result in results:
            print(
        f"score={result['score']:.4f} | "
        f"text={result['text'][:100]}"
    )
        
        return results
        
    