

class RAGPipeline:
    
    def __init__(self,retriever,llm_service):
        self.retriver=retriever
        self.llm_service=llm_service
        
        
    def ask(self,collection_name: str, query:str,top_k=4):
        
        
        #retriver
        
        results=self.retriver(collection_name=collection_name,query=query,top_k=top_k)
        
        if not results:
            return ({"answer":"I could not find the relevant information in the doucments"
                ,"sources":[]
            })
        context="\n\n".join(results['text'] for result in results)
        
        prompt=f"""
            You are a helpful assistant answering questions about a document.

             Use ONLY the information provided in the context.
             
             Context:
             {context}
             
             Question:
             {query}
             
             If the answer cannot be found in the context,
             say that the information is not available in the document.
             
             Answer:
             """
        answer=self.llm_service.generate(prompt)
        
        return {
            "Answer":answer,
            "Sources":results
         }
        
        
        