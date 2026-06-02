import os  
import time  
import shutil  
import tempfile  
from fastapi import FastAPI, UploadFile, File, HTTPException  
from models import QueryRequest  
from pageindex import PageIndexClient  
from dotenv import load_dotenv

from vectorless_single import vectorless_rag

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")

app = FastAPI(title="Vectorless RAG API")  
pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

active_doc = {  
    "doc_id": None,  
    "tree": None  
}


@app.post("/upload")  
def upload_document(file: UploadFile = File(...)):  
    # ✅ Use tempfile for cross-platform compatibility  
    suffix = os.path.splitext(file.filename)[1]  # e.g. ".pdf"  
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:  
        shutil.copyfileobj(file.file, tmp)  
        temp_path = tmp.name  # Windows-safe path like C:\Users\...\AppData\Local\Temp\xxx.pdf

    try:  
        result = pi_client.submit_document(temp_path)  
        doc_id = result["doc_id"]  
    except Exception as e:  
        raise HTTPException(status_code=500, detail=f"PageIndex submission failed: {e}")  
    finally:  
        os.remove(temp_path)  # Clean up

    # Poll until processing completes  
    max_retries = 60  
    for _ in range(max_retries):  
        status_result = pi_client.get_document(doc_id)  
        status = status_result["status"]

        if status == "completed":  
            break  
        elif status == "failed":  
            raise HTTPException(status_code=500, detail="PageIndex document processing failed.")

        time.sleep(2)  
    else:  
        raise HTTPException(status_code=504, detail="Document processing timed out.")

    tree_result = pi_client.get_tree(doc_id, node_summary=True)  
    active_doc["doc_id"] = doc_id  
    active_doc["tree"] = tree_result.get("result", [])

    return {  
        "message": "Document uploaded and processed successfully.",  
        "doc_id": doc_id,  
        "total_sections": len(active_doc["tree"])  
    }
@app.post("/query")  
def handle_query(query: QueryRequest):  
    if active_doc["tree"] is None:  
        raise HTTPException(  
            status_code=400,  
            detail="No document uploaded yet. Please upload a file first via /upload."  
        )

    from vectorless_single import vectorless_rag  
    answer = vectorless_rag(query.query, active_doc["tree"], verbose=False)

    return {"answer": answer, "doc_id": active_doc["doc_id"]}  
