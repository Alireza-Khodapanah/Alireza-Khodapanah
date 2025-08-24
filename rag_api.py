import nest_asyncio
nest_asyncio.apply()

import os
import hashlib
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from unsloth import FastModel
import torch
from transformers import TextStreamer

# Initialize FastAPI app
app = FastAPI(
    title="RAG API with Gemma & FastAPI",
    version="1.0.0"
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    context: str

class DeleteRequest(BaseModel):
    filename: str

class DeleteResponse(BaseModel):
    status: str
    message: str
    deleted_filename: str

# Global variables
index = None
retriever = None
text_splitter = None
model = None
tokenizer = None

# Initialize components
def initialize_components():
    global index, retriever, text_splitter, model, tokenizer
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name="heydariAI/persian-embeddings")
    Settings.embed_model = embed_model
    Settings.llm = None
    Settings.chunk_size = 2048
    Settings.chunk_overlap = 512
    
    # Initialize text splitter
    text_splitter = SentenceSplitter(chunk_size=4096, chunk_overlap=1024)
    
    # Load initial documents
    docs = SimpleDirectoryReader(input_dir="./data", filename_as_id=True).load_data()
    nodes = text_splitter.get_nodes_from_documents(docs)
    
    print("Creating the vector index... This may take a moment.")
    index = VectorStoreIndex(nodes)
    print("Vector index created successfully!")
    
    retriever = index.as_retriever(similarity_top_k=5)
    
    # Initialize model (you'll need to uncomment and configure this with your HF token)
    # hf_token = "your_hf_token_here"  # Replace with your actual token
    # model_id = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
    # model, tokenizer = FastModel.from_pretrained(
    #     model_name=model_id,
    #     dtype=None,
    #     max_seq_length=20000,
    #     load_in_4bit=True,
    #     token=hf_token,
    # )

# Prompt template
PROMPT_TEMPLATE = """
<start_of_turn>user
**دستورالعمل:** به سوال زیر فقط و فقط بر اساس متن ارائه شده پاسخ بده.
اگر پاسخ در متن موجود نیست، به صراحت بگو "پاسخ در متون ارائه شده یافت نشد." و از ارائه هرگونه اطلاعات اضافی خودداری کن.

**متن:**
{context}

**سوال:**
{question}
<end_of_turn>
<start_of_turn>model
"""

# Helper functions
def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()

def is_hash_in_index(file_hash: str, index: VectorStoreIndex) -> bool:
    """Check if a file with the given hash already exists in the index."""
    for doc in index.docstore.docs.values():
        if doc.metadata.get("file_hash") == file_hash:
            return True
    return False

def find_document_by_filename(filename: str, index: VectorStoreIndex) -> str:
    """Find document ID by filename in the index."""
    for doc_id, doc in index.docstore.docs.items():
        if doc.metadata.get("file_name") == filename or doc_id == filename:
            return doc_id
    return None

def get_nodes_by_document_id(doc_id: str, index: VectorStoreIndex) -> List[str]:
    """Get all node IDs that belong to a specific document."""
    node_ids = []
    for node_id, node in index.docstore.docs.items():
        if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id:
            node_ids.append(node_id)
        elif node_id.startswith(doc_id):
            node_ids.append(node_id)
    return node_ids

def generate_api_response(prompt: str, max_new_tokens: int = 512) -> str:
    """Generate response using the model."""
    if model is None or tokenizer is None:
        return "Model not initialized. Please configure the model first."
    
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.01,
        top_p=0.95,
        top_k=50,
    )

    response_text = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response_text.strip()

# API Endpoints
@app.on_event("startup")
async def startup_event():
    initialize_components()

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    global retriever

    print(f"Received query: {request.question}")

    retrieved_nodes = retriever.retrieve(request.question)
    context = "\n\n---\n\n".join([node.get_content() for node in retrieved_nodes])

    formatted_prompt = PROMPT_TEMPLATE.format(context=context, question=request.question)
    answer = generate_api_response(formatted_prompt)

    print(f"Generated answer: {answer}")

    return QueryResponse(answer=answer, context=context)

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    global retriever, index, text_splitter
    
    processed_files, skipped_files, new_files_added = [], [], False
    
    try:
        os.makedirs("./data", exist_ok=True)
        
        for file in files:
            clean_filename = os.path.basename(file.filename)
            content = await file.read()

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

@app.delete("/delete", response_model=DeleteResponse)
async def delete_file(request: DeleteRequest):
    """Delete a file from both the vector index and the filesystem."""
    global retriever, index
    
    try:
        filename = request.filename
        
        # Find the document in the index
        doc_id = find_document_by_filename(filename, index)
        
        if doc_id is None:
            raise HTTPException(
                status_code=404, 
                detail=f"File '{filename}' not found in the index."
            )
        
        # Get all nodes associated with this document
        node_ids = get_nodes_by_document_id(doc_id, index)
        
        # Delete from vector store
        if doc_id in index.docstore.docs:
            index.docstore.delete_document(doc_id)
        
        # Delete associated nodes
        for node_id in node_ids:
            if node_id in index.docstore.docs:
                index.docstore.delete_document(node_id)
        
        # Remove from vector store index
        try:
            if hasattr(index, 'vector_store') and hasattr(index.vector_store, 'delete'):
                index.vector_store.delete(doc_id)
            elif hasattr(index, '_vector_store') and hasattr(index._vector_store, 'delete'):
                index._vector_store.delete(doc_id)
        except Exception as e:
            print(f"Warning: Could not delete from vector store: {e}")
        
        # Delete from filesystem
        file_path = os.path.join("./data", filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            filesystem_deleted = True
        else:
            filesystem_deleted = False
        
        # Refresh the retriever
        retriever = index.as_retriever(similarity_top_k=5)
        
        message = f"File '{filename}' deleted from index"
        if filesystem_deleted:
            message += " and filesystem"
        else:
            message += " (file not found in filesystem)"
        
        return DeleteResponse(
            status="success",
            message=message,
            deleted_filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.get("/files")
async def list_files():
    """List all files in the index and filesystem."""
    global index
    
    try:
        # Files in index
        index_files = []
        for doc_id, doc in index.docstore.docs.items():
            if hasattr(doc, 'metadata') and 'file_name' in doc.metadata:
                index_files.append({
                    "doc_id": doc_id,
                    "filename": doc.metadata['file_name'],
                    "file_hash": doc.metadata.get('file_hash', 'N/A')
                })
        
        # Files in filesystem
        filesystem_files = []
        data_dir = "./data"
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)
                if os.path.isfile(file_path):
                    filesystem_files.append(filename)
        
        return {
            "index_files": index_files,
            "filesystem_files": filesystem_files,
            "total_index_files": len(index_files),
            "total_filesystem_files": len(filesystem_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)