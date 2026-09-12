from app.retrieval.baseretreiver import BaseRetriever


class BM25Retriever(BaseRetriever):

    def __init__(self, bm25_store):
        self.bm25_store = bm25_store

    def query_retriever(
        self,
        collection_name: str,
        query: str,
        top_k: int = 4
    ):
        data = self.bm25_store.load_index(collection_name)

        bm25 = data["bm25"]
        documents = data["documents"]

        tokenized_query = query.lower().split()

        scores = bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "text": documents[index].page_content,
                "score": float(scores[index]),
                "metadata": documents[index].metadata
            })

        return results