from rank_bm25 import BM25Okapi
from app.retrieval.baseretreiver import BaseRetriever


class BM25Retriever(BaseRetriever):

    def __init__(self, documents):
        self.documents = documents

        tokenized_documents = [
            document.page_content.lower().split()
            for document in documents
        ] 

        self.bm25 = BM25Okapi(tokenized_documents)

    def query_retriever(self, query: str, top_k: int = 4):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "text": self.documents[index].page_content,
                "score": float(scores[index]),
                "metadata": self.documents[index].metadata
            })

        return results