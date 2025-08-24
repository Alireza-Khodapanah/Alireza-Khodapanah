import hashlib
from typing import List, Dict, Any
from fastapi import UploadFile, File, HTTPException
import shutil
import os
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document

# این متغیرهای سراسری باید جایی در کد شما تعریف شده باشند
global index
global retriever
global text_splitter

# --- توابع کمکی اصلاح شده ---

def calculate_file_hash(content: bytes) -> str:
    """هش SHA-256 محتوای یک فایل (که به صورت بایت ارائه شده) را محاسبه می‌کند."""
    return hashlib.sha256(content).hexdigest()

def is_hash_in_index(file_hash: str, index: VectorStoreIndex) -> bool:
    """بررسی می‌کند آیا فایلی با هش مشخص شده قبلاً در فراداده ایندکس وجود دارد یا خیر."""
    for doc in index.docstore.docs.values():
        if doc.metadata.get("file_hash") == file_hash:
            return True
    return False

def delete_file_from_index(filename: str, index: VectorStoreIndex) -> bool:
    """حذف فایل از ایندکس بر اساس نام فایل"""
    deleted = False
    docs_to_remove = []
    
    for doc_id, doc in index.docstore.docs.items():
        if doc.metadata.get("file_name") == filename:
            docs_to_remove.append(doc_id)
            deleted = True
    
    # حذف داکیومنت‌ها از ایندکس
    for doc_id in docs_to_remove:
        if doc_id in index.docstore.docs:
            del index.docstore.docs[doc_id]
    
    # بازسازی ایندکس
    if deleted:
        # حذف نودهای مربوطه
        nodes_to_remove = []
        for node_id, node in index.docstore.nodes.items():
            if node.metadata.get("file_name") == filename:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            if node_id in index.docstore.nodes:
                del index.docstore.nodes[node_id]
    
    return deleted

# --- اندپوینت آپلود نهایی ---

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    global retriever
    processed_files, skipped_files, new_files_added = [], [], False
    try:
        os.makedirs("./data", exist_ok=True)
        for file in files:
            clean_filename = os.path.basename(file.filename)
            content = await file.read()

            # حالا این فراخوانی با تابع اصلاح شده کاملاً مطابقت دارد
            file_hash = calculate_file_hash(content)

            if is_hash_in_index(file_hash, index):
                skipped_files.append({"filename": clean_filename, "reason": "Duplicate content."})
                continue
            if clean_filename in index.docstore.docs:
                skipped_files.append({"filename": clean_filename, "reason": "Duplicate name."})
                continue

            save_path = os.path.join("./data", clean_filename)
            with open(save_path, "wb") as f:
                f.write(content)

            loaded_docs = SimpleDirectoryReader(input_files=[save_path]).load_data()

            doc_with_correct_id = Document(
                id_=clean_filename,
                text=" ".join([d.get_content() for d in loaded_docs]),
                metadata={"file_hash": file_hash, "file_name": clean_filename}
            )
            doc_with_correct_id.excluded_embed_metadata_keys.extend(["file_name", "file_hash"])

            new_nodes = text_splitter.get_nodes_from_documents([doc_with_correct_id])
            index.insert_nodes(new_nodes)

            new_files_added = True
            processed_files.append({"filename": clean_filename, "chunks_added": len(new_nodes)})

        if new_files_added:
            retriever = index.as_retriever(similarity_top_k=5)

        return {"status": "success", "files_processed": processed_files, "files_skipped": skipped_files}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- اندپوینت حذف فایل ---

@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    global retriever
    
    try:
        # بررسی وجود فایل در سیستم فایل
        file_path = os.path.join("./data", filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
        
        # حذف از ایندکس
        deleted_from_index = delete_file_from_index(filename, index)
        
        # حذف از سیستم فایل
        os.remove(file_path)
        
        # بازسازی retriever
        if deleted_from_index:
            retriever = index.as_retriever(similarity_top_k=5)
        
        return {
            "status": "success",
            "message": f"File '{filename}' deleted successfully",
            "deleted_from_index": deleted_from_index,
            "deleted_from_filesystem": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

# --- اندپوینت لیست فایل‌ها ---

@app.get("/files")
async def list_files():
    """لیست تمام فایل‌های موجود در ایندکس"""
    try:
        files = []
        for doc_id, doc in index.docstore.docs.items():
            if doc.metadata.get("file_name"):
                files.append({
                    "filename": doc.metadata.get("file_name"),
                    "file_hash": doc.metadata.get("file_hash"),
                    "doc_id": doc_id
                })
        
        return {
            "status": "success",
            "total_files": len(files),
            "files": files
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")