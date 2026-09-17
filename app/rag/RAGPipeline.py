

class RAGPipeline:
    
    def __init__(self, retriever, llm_service):
        self.retriever = retriever
        self.retriver = retriever  # alias for backward compatibility
        self.llm_service = llm_service
        
    def ask(self, collection_name: str, query: str, top_k: int = 5) -> dict:
        results = self.retriever.query_retriever(
            collection_name=collection_name,
            query=query,
            top_k=top_k
        )
        
        if not results:
            return {
                "Answer": "I could not find the relevant information in the documents.",
                "answer": "I could not find the relevant information in the documents.",
                "Sources": [],
                "sources": [],
            }
            
        context = "\n\n".join(result["text"] for result in results if "text" in result)
        
        prompt = f"""You are a helpful assistant answering questions about a document.

Use ONLY the information provided in the context.

Context:
{context}

Question:
{query}

If the answer cannot be found in the context, say that the information is not available in the document.

Answer:"""

        answer = self.llm_service.generate(prompt)
        
        return {
            "Answer": answer,
            "answer": answer,
            "Sources": results,
            "sources": results,
        }