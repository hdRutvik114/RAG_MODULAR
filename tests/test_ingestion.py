from app.ingestion.pipeline import  IngestionPipeline
from app.ingestion.splitter import DocumentSplitter 
from app.ingestion.loader import PDFLoader
from app.embeddings.embeddings import EmbeddingService

loader=PDFLoader()

splitter=DocumentSplitter()

pipeline=IngestionPipeline(loader,splitter)

chunks=pipeline.process("data/documents/attention_paper.pdf")

# for i,chunk in enumerate(chunks):
    
#     if(i==3):
#         break
#     print(f"\nchunk------ {i}------")
#     print(chunk.page_content[:300])
#     print(f"Metadata:{chunk.metadata}")


texts=[chunk.page_content for chunk in chunks]
print("-"*15)
print(texts)

embedder=EmbeddingService()
embedder.embed_doucments(texts)
query=embedder.embed_query("HI bro ")
print(query)