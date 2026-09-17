import logging
from app.retrieval.baseretreiver import BaseRetriever


logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):

    def __init__(self, bm25_store):
        self.bm25_store = bm25_store

    def query_retriever(
        self,
        collection_name: str,
        query: str,
        top_k: int = 4
    ):
        logger.info("BM25 query: collection=%s top_k=%d", collection_name, top_k)
        data = self.bm25_store.load_index(collection_name)

        bm25 = data["bm25"]
        documents = data["documents"]

        tokenized_query = query.lower().split()
        logger.debug("Tokenized query: %s", tokenized_query)

        scores = bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]
        logger.info("BM25 top indices: %s", ranked_indices.tolist() if hasattr(ranked_indices, 'tolist') else list(ranked_indices))

        results = []

        for index in ranked_indices:
            results.append({
                "text": documents[index].page_content,
                "score": float(scores[index]),
                "metadata": documents[index].metadata
            })

        logger.info("BM25 returned %d results for collection=%s", len(results), collection_name)
        return results