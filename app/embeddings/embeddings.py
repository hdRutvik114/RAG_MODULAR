from sentence_transformers import SentenceTransformer



class EmbeddingService:
    
    def __init__(self,model_name : str="all-MiniLM-L6-v2"):
        self.model_name=model_name
        self.model=None
        self._loadmodel()
        
        
    def _loadmodel(self):
        try:
            print(f"Bringing the model{self.model_name}")
            self.model=SentenceTransformer(self.model_name)
            print(f"MOdel loader successfully:{self.model.get_embedding_dimension()}")
            
        except Exception as e:
            print(f"The model didnt load : {e}")
            raise
        
    def embed_doucments(self,texts:list[str])->list:
        if self.model is None:
            raise ValueError("No model Found bro")
        try:
            embeddings=self.model.encode(texts,show_progress_bar=True)
            print(f"Length of documents{len(embeddings)}")
            return embeddings
        except Exception as e:
            print(f"Error while embeddings{e}")
            raise
        
    def embed_query(self,query:str):
        if not self.model:
            raise ValueError("Ebedding model is loaded")
        
        return self.model.encode(query)
    
    