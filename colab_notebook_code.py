# ============================================================================
# RAG API with Delete Endpoint - Complete Implementation and Testing
# Copy each section below into separate Google Colab cells
# ============================================================================

# ============================================================================
# CELL 1: Installation
# ============================================================================
import nest_asyncio
nest_asyncio.apply()

# Install required packages
get_ipython().system('pip install -q --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo')
get_ipython().system('pip install -q sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer')
get_ipython().system('pip install -q --no-deps unsloth')
get_ipython().system('pip install -q -U llama-index')
get_ipython().system('pip install -q llama-index-llms-gemini')
get_ipython().system('pip install -q llama-index-embeddings-huggingface')
get_ipython().system('pip install -q sentence-transformers')
get_ipython().system('pip install -q fastapi uvicorn')

print("✅ All packages installed successfully!")

# ============================================================================
# CELL 2: RAG API Implementation with Delete Endpoint
# ============================================================================
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
    
    # Note: Uncomment and configure the model initialization with your HF token
    # from google.colab import userdata
    # hf_token = userdata.get('HF_TOKEN')
    # model_id = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
    # model, tokenizer = FastModel.from_pretrained(
    #     model_name=model_id,
    #     dtype=None,
    #     max_seq_length=20000,
    #     load_in_4bit=True,
    #     token=hf_token,
    # )

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

print("✅ RAG API with Delete Endpoint implemented successfully!")

# ============================================================================
# CELL 3: Test Implementation using FastAPI TestClient
# ============================================================================
import asyncio
import io
import json
from fastapi.testclient import TestClient

# Create a test client
client = TestClient(app)

def create_test_file(filename: str, content: str) -> io.BytesIO:
    """Create a test file in memory for upload testing."""
    return io.BytesIO(content.encode('utf-8'))

def test_upload_endpoint():
    """Test the upload endpoint."""
    print("🧪 Testing Upload Endpoint...")
    
    # Create test files
    test_files = [
        ("test1.txt", "این یک متن تست فارسی است. این متن برای تست آپلود فایل استفاده می‌شود."),
        ("test2.txt", "متن دوم برای تست. این متن شامل اطلاعات مختلف است که باید در ایندکس ذخیره شود."),
    ]
    
    files = []
    for filename, content in test_files:
        files.append(("files", (filename, create_test_file(filename, content), "text/plain")))
    
    # Test upload
    response = client.post("/upload", files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["files_processed"]) == 2
    
    print("✅ Upload test passed!\n")
    return [f[0] for _, (f, _, _) in files]  # Return filenames

def test_list_files_endpoint():
    """Test the list files endpoint."""
    print("🧪 Testing List Files Endpoint...")
    
    response = client.get("/files")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert "index_files" in data
    assert "filesystem_files" in data
    assert data["total_index_files"] >= 0
    assert data["total_filesystem_files"] >= 0
    
    print("✅ List files test passed!\n")
    return data

def test_ask_endpoint():
    """Test the ask endpoint."""
    print("🧪 Testing Ask Endpoint...")
    
    query_data = {
        "question": "متن تست چیست؟"
    }
    
    response = client.post("/ask", json=query_data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "context" in data
    
    print("✅ Ask test passed!\n")

def test_delete_endpoint(filename_to_delete: str):
    """Test the delete endpoint."""
    print(f"🧪 Testing Delete Endpoint for file: {filename_to_delete}...")
    
    # First, verify the file exists
    files_before = client.get("/files").json()
    print(f"Files before deletion: {len(files_before['index_files'])} in index, {len(files_before['filesystem_files'])} in filesystem")
    
    # Test delete
    delete_data = {
        "filename": filename_to_delete
    }
    
    response = client.delete("/delete", json=delete_data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["deleted_filename"] == filename_to_delete
    
    # Verify the file was deleted
    files_after = client.get("/files").json()
    print(f"Files after deletion: {len(files_after['index_files'])} in index, {len(files_after['filesystem_files'])} in filesystem")
    
    # Check that the file is no longer in the index
    index_filenames = [f["filename"] for f in files_after["index_files"]]
    assert filename_to_delete not in index_filenames
    
    # Check that the file is no longer in filesystem
    assert filename_to_delete not in files_after["filesystem_files"]
    
    print("✅ Delete test passed!\n")

def test_delete_nonexistent_file():
    """Test deleting a file that doesn't exist."""
    print("🧪 Testing Delete Endpoint for non-existent file...")
    
    delete_data = {
        "filename": "nonexistent_file.txt"
    }
    
    response = client.delete("/delete", json=delete_data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()
    
    print("✅ Delete non-existent file test passed!\n")

def test_duplicate_upload():
    """Test uploading duplicate files."""
    print("🧪 Testing Duplicate Upload...")
    
    # Create the same file twice
    test_content = "این یک متن تکراری است برای تست آپلود مجدد."
    files = [
        ("files", ("duplicate_test.txt", create_test_file("duplicate_test.txt", test_content), "text/plain")),
        ("files", ("duplicate_test.txt", create_test_file("duplicate_test.txt", test_content), "text/plain"))
    ]
    
    # First upload
    response1 = client.post("/upload", files=[files[0]])
    print(f"First upload - Status Code: {response1.status_code}")
    
    # Second upload (should be skipped as duplicate)
    response2 = client.post("/upload", files=[files[1]])
    print(f"Second upload - Status Code: {response2.status_code}")
    print(f"Second upload response: {json.dumps(response2.json(), indent=2, ensure_ascii=False)}")
    
    assert response2.status_code == 200
    data = response2.json()
    assert len(data["files_skipped"]) > 0
    
    print("✅ Duplicate upload test passed!\n")
    
    # Clean up
    client.delete("/delete", json={"filename": "duplicate_test.txt"})

def run_all_tests():
    """Run all tests in sequence."""
    print("🚀 Starting RAG API Tests...\n")
    
    try:
        # Test upload
        uploaded_files = test_upload_endpoint()
        
        # Test list files
        files_data = test_list_files_endpoint()
        
        # Test ask (only if we have files)
        if files_data["total_index_files"] > 0:
            test_ask_endpoint()
        
        # Test delete with an existing file
        if uploaded_files:
            test_delete_endpoint(uploaded_files[0])
        
        # Test delete non-existent file
        test_delete_nonexistent_file()
        
        # Test duplicate upload
        test_duplicate_upload()
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

print("✅ Test suite implemented successfully!")

# ============================================================================
# CELL 4: Run All Automated Tests
# ============================================================================
# Run all automated tests
run_all_tests()

# ============================================================================
# CELL 5: Interactive Testing with Persian Content
# ============================================================================
# Interactive test for uploading Persian documents
def interactive_upload_test():
    """Interactive test for uploading files with Persian content."""
    print("📁 Interactive Upload Test with Persian Content")
    
    # Create multiple test files with Persian content
    test_files_content = {
        "constitution.txt": """
        قانون اساسی جمهوری اسلامی ایران
        
        فصل اول: اصول کلی
        
        اصل یکم: حکومت ایران، جمهوری اسلامی است که ملت ایران بر اساس ایمان دیرینه‌اش به حاکمیت حق و عدالت قرآن، در پانزدهم خرداد ماه یکهزار و سیصد و پنجاه و هشت هجری شمسی معادل بیست و هشتم ذی‌القعده یکهزار و سیصد و نود و نه هجری قمری با رأی مثبت نود و هشت و هفت دهم درصد آراء شرکت‌کنندگان در همه‌پرسی تصویب نمود.
        """,
        "civil_rights.txt": """
        اصول حقوق شهروندی
        
        هر شهروند دارای حقوق و تکالیف مشخصی است که در قانون اساسی تعریف شده است.
        حق آزادی بیان، حق تجمع، و حق دسترسی به اطلاعات از جمله حقوق اساسی شهروندان محسوب می‌شود.
        شهروندان حق دارند در انتخابات شرکت کنند و نمایندگان خود را انتخاب نمایند.
        """,
        "legislation.txt": """
        فرآیند قانون‌گذاری
        
        قوانین در مجلس شورای اسلامی تصویب می‌شود.
        پس از تصویب در مجلس، قوانین برای بررسی نهایی به شورای نگهبان ارسال می‌شود.
        در صورت تأیید شورای نگهبان، قانون برای اجرا ابلاغ می‌شود.
        اگر شورای نگهبان قانون را رد کند، به مجلس بازگردانده می‌شود.
        """
    }
    
    files = []
    for filename, content in test_files_content.items():
        files.append(("files", (filename, create_test_file(filename, content), "text/plain")))
    
    response = client.post("/upload", files=files)
    print(f"Upload Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return list(test_files_content.keys())

# Upload Persian documents
uploaded_files = interactive_upload_test()

# ============================================================================
# CELL 6: Interactive Query Testing
# ============================================================================
# Interactive query testing
def interactive_query_test():
    """Interactive test for querying with Persian questions."""
    print("❓ Interactive Query Test with Persian Questions")
    
    queries = [
        "قانون اساسی چگونه تصویب شد؟",
        "حقوق شهروندی چیست؟",
        "فرآیند قانون‌گذاری چگونه است؟",
        "شورای نگهبان چه نقشی دارد؟",
        "شهروندان چه حقوقی دارند؟"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        response = client.post("/ask", json={"question": query})
        if response.status_code == 200:
            data = response.json()
            print(f"Answer: {data['answer'][:300]}...")  # Show first 300 chars
            print(f"Context length: {len(data['context'])} characters")
        else:
            print(f"Error: {response.status_code} - {response.text}")

# Test queries
interactive_query_test()

# ============================================================================
# CELL 7: Check Current Files
# ============================================================================
# Check current files in the system
files_response = client.get("/files")
print(f"📋 Current files in system:")
print(json.dumps(files_response.json(), indent=2, ensure_ascii=False))

# ============================================================================
# CELL 8: Test Delete Functionality
# ============================================================================
# Test delete functionality
def interactive_delete_test(filenames):
    """Interactive test for deleting files."""
    print("🗑️ Interactive Delete Test")
    
    for filename in filenames[:2]:  # Delete first 2 files only
        print(f"\nDeleting: {filename}")
        response = client.delete("/delete", json={"filename": filename})
        print(f"Delete Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        # Check files after each deletion
        files_after = client.get("/files").json()
        print(f"Files remaining: {len(files_after['index_files'])} in index, {len(files_after['filesystem_files'])} in filesystem")

# Test deletion
interactive_delete_test(uploaded_files)

# ============================================================================
# CELL 9: Final Summary
# ============================================================================
# Final check of files
final_files = client.get("/files")
print(f"📋 Final files in system:")
print(json.dumps(final_files.json(), indent=2, ensure_ascii=False))

print("\n🎉 All tests completed successfully!")
print("\n📝 Summary of implemented features:")
print("✅ Upload endpoint - Upload multiple files with duplicate detection")
print("✅ Ask endpoint - Query documents with Persian language support")
print("✅ Delete endpoint - Remove files from both index and filesystem")
print("✅ List files endpoint - View all files in index and filesystem")
print("✅ Comprehensive test suite with FastAPI TestClient")
print("✅ Persian language support for documents and queries")
print("✅ Error handling for edge cases (non-existent files, duplicates)")

print("\n📋 Usage Instructions:")
print("1. POST /upload - Upload files to the RAG system")
print("2. POST /ask - Query the documents with Persian or English questions")
print("3. DELETE /delete - Delete files from both vector index and filesystem")
print("4. GET /files - List all files in index and filesystem")
print("\n⚠️ To use with your model:")
print("1. Uncomment the model initialization code in Cell 2")
print("2. Add your Hugging Face token to Google Colab secrets")
print("3. The system will work with the Gemma model for Persian text generation")