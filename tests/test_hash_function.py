from app.utils.hash_utils import get_pdf_collection_name

pdf_path = "data/documents/IgnitedMinds.pdf"#ignite
pdf_path="data/documents/attention_paper.pdf"

collection_name = get_pdf_collection_name(pdf_path)

print(collection_name)
