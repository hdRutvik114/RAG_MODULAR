import os
import pickle

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Store:

    def __init__(self, base_path: str = "data/bm25"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _get_index_path(self, collection_name: str) -> str:
        return os.path.join(
            self.base_path,
            f"{collection_name}.pkl"
        )

    def index_exists(self, collection_name: str) -> bool:
        return os.path.exists(
            self._get_index_path(collection_name)
        )

    def create_index(
        self,
        collection_name: str,
        documents: list[Document]
    ):
        tokenized_documents = [
            document.page_content.lower().split()
            for document in documents
        ]

        bm25 = BM25Okapi(tokenized_documents)

        data = {
            "bm25": bm25,
            "documents": documents
        }

        with open(
            self._get_index_path(collection_name),
            "wb"
        ) as file:
            pickle.dump(data, file)

    def load_index(self, collection_name: str):
        with open(
            self._get_index_path(collection_name),
            "rb"
        ) as file:
            return pickle.load(file)