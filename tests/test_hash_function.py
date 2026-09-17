from app.utils.hash_utils import get_pdf_collection_name
from pathlib import Path


def test_get_pdf_collection_name():
    pdf_path = Path("data/documents/attention_paper.pdf")
    if not pdf_path.exists():
        pdf_path = next(Path("data/documents").glob("*.pdf"))

    collection_name = get_pdf_collection_name(str(pdf_path))
    assert collection_name.startswith("doc_")
    print(f"Collection name: {collection_name}")


if __name__ == "__main__":
    test_get_pdf_collection_name()
