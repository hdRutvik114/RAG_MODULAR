import hashlib
import os
import re

def get_pdf_collection_name(pdf_path:str)->str:
    
    hasher=hashlib.sha256()
    
    with open(pdf_path,"rb") as f:
        for chunk in iter(lambda: f.read(4096),b""):
            hasher.update(chunk)
            
    
    file_hash=hasher.hexdigest()[:12]
    
    base_name=os.path.splitext(
        os.path.basename(pdf_path)
    )[0]
    
    clean_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        base_name
    ).lower()

    
    return f"doc_{clean_name[:20]}_{file_hash}"