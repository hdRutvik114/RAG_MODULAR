from app.retrieval.baseretreiver import BaseRetriever


class HybridRetriever(BaseRetriever):

    def __init__(self, vector_retriever, bm25_retriever, rrf_k=60):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def query_retriever(
        self,
        collection_name: str,
        query: str,
        top_k: int = 4
    ):

        # 1. Vector search
        vector_results = self.vector_retriever.query_retriever(
            collection_name=collection_name,
            query=query,
            top_k=top_k
        )

        # 2. BM25 search
        bm25_results = self.bm25_retriever.query_retriever(
            query=query,
            top_k=top_k
        )

        # 3. Store RRF scores
        rrf_scores = {}
        documents = {}

        # Vector ranking
        for rank, result in enumerate(vector_results, start=1):

            doc_key = result["text"]

            rrf_scores[doc_key] = (
                rrf_scores.get(doc_key, 0)
                + 1 / (self.rrf_k + rank)
            )

            documents[doc_key] = result

        # BM25 ranking
        for rank, result in enumerate(bm25_results, start=1):

            doc_key = result["text"]

            rrf_scores[doc_key] = (
                rrf_scores.get(doc_key, 0)
                + 1 / (self.rrf_k + rank)
            )

            documents[doc_key] = result

        # 4. Sort by RRF score
        ranked_documents = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 5. Build final results
        final_results = []

        for doc_key, rrf_score in ranked_documents[:top_k]:

            result = documents[doc_key].copy()

            result["rrf_score"] = rrf_score

            final_results.append(result)

        return final_results