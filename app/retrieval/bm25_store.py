import os
import pickle
from io import BytesIO
import logging

try:
    import boto3
except Exception:
    boto3 = None

logger = logging.getLogger(__name__)

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

        index_path = self._get_index_path(collection_name)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        # write locally
        with open(index_path, "wb") as file:
            pickle.dump(data, file)

        logger.info("Created BM25 index for collection=%s at %s (documents=%d)", collection_name, index_path, len(documents))

        # if S3 configured, upload the pickle there as well (best-effort)
        bucket = os.getenv("S3_BUCKET")
        if bucket and boto3 is not None:
            try:
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name=os.getenv("AWS_REGION"),
                )
                key = f"bm25/{collection_name}.pkl"
                with open(index_path, "rb") as f:
                    s3.upload_fileobj(f, bucket, key)
                logger.info("Uploaded BM25 index for collection=%s to s3://%s/%s", collection_name, bucket, key)
            except Exception as e:
                logger.warning("Failed to upload BM25 index for collection=%s to S3: %s", collection_name, e)

    def load_index(self, collection_name: str):
        index_path = self._get_index_path(collection_name)
        # prefer local copy
        if os.path.exists(index_path):
            logger.info("Loading BM25 index for collection=%s from local path %s", collection_name, index_path)
            with open(index_path, "rb") as file:
                data = pickle.load(file)
                logger.info("Loaded BM25 index for collection=%s (documents=%d)", collection_name, len(data.get('documents', [])))
                return data

        # otherwise try S3
        bucket = os.getenv("S3_BUCKET")
        if bucket and boto3 is not None:
            try:
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name=os.getenv("AWS_REGION"),
                )
                key = f"bm25/{collection_name}.pkl"
                bio = BytesIO()
                s3.download_fileobj(bucket, key, bio)
                bio.seek(0)
                data = pickle.load(bio)
                logger.info("Loaded BM25 index for collection=%s from S3 s3://%s/%s (documents=%d)", collection_name, bucket, key, len(data.get('documents', [])))
                return data
            except Exception as e:
                logger.warning("Failed to load BM25 index for collection=%s from S3: %s", collection_name, e)

        raise FileNotFoundError(f"BM25 index not found for collection: {collection_name}")