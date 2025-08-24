# RAG API with Gemma & FastAPI

A Retrieval-Augmented Generation (RAG) system built with FastAPI, using the Gemma model and Persian embeddings for question answering.

## Features

- ✅ **File Upload**: Upload documents to the knowledge base
- ✅ **Question Answering**: Ask questions and get AI-generated answers based on uploaded content
- ✅ **File Management**: List all files in the system
- 🆕 **File Deletion**: Remove files from both the filesystem and vector index
- 🔍 **Smart Retrieval**: Uses similarity search to find relevant context
- 🌐 **Persian Language Support**: Optimized for Persian text with specialized embeddings

## New Delete Endpoint

### DELETE `/delete/{filename}`

Deletes a file from both the filesystem and the vector index.

**Parameters:**
- `filename` (path parameter): Name of the file to delete

**Response:**
```json
{
  "status": "success",
  "message": "File 'filename.txt' deleted successfully",
  "deleted_from_index": true,
  "deleted_from_filesystem": true
}
```

**Error Responses:**
- `404 Not Found`: File doesn't exist
- `500 Internal Server Error`: Server error during deletion

**Example Usage:**
```bash
# Delete a file
curl -X DELETE "http://localhost:8000/delete/test_document.txt"

# Response
{
  "status": "success",
  "message": "File 'test_document.txt' deleted successfully",
  "deleted_from_index": true,
  "deleted_from_filesystem": true
}
```

## API Endpoints

### 1. Upload Files
- **POST** `/upload`
- Upload multiple files to the knowledge base
- Supports duplicate detection (by content hash and filename)

### 2. Ask Questions
- **POST** `/ask`
- Ask questions and get AI-generated answers
- Uses RAG to provide context-aware responses

### 3. List Files
- **GET** `/files`
- List all files currently in the system
- Shows file metadata including hashes

### 4. Delete Files
- **DELETE** `/delete/{filename}`
- Remove files from the system
- Cleans up both filesystem and vector index

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -q --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
pip install -q sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
pip install -q --no-deps unsloth
pip install -q -U llama-index
pip install -q llama-index-llms-gemini
pip install -q llama-index-embeddings-huggingface
pip install -q sentence-transformers
pip install -q fastapi uvicorn
```

### 2. Set Environment Variables
```python
from google.colab import userdata
hf_token = userdata.get('HF_TOKEN')  # Your HuggingFace token
```

### 3. Run the Application
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Testing

### Using the Test Script

1. **Run the full test suite:**
```python
from test_colab import test_api_endpoints
test_api_endpoints()
```

2. **Run a quick test:**
```python
from test_colab import quick_test
quick_test()
```

### Manual Testing

1. **Start the server:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

2. **Test file upload:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document1.txt" \
  -F "files=@document2.txt"
```

3. **Test question answering:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the documents?"}'
```

4. **Test file listing:**
```bash
curl "http://localhost:8000/files"
```

5. **Test file deletion:**
```bash
curl -X DELETE "http://localhost:8000/delete/document1.txt"
```

## File Management Workflow

### 1. Upload Documents
- Documents are processed and chunked
- Content is embedded using Persian embeddings
- Files are stored in the `./data` directory
- Vector index is updated with new content

### 2. Query Documents
- Questions are processed through the RAG pipeline
- Relevant context is retrieved using similarity search
- Gemma model generates answers based on retrieved context

### 3. Delete Documents
- Files are removed from the filesystem
- Documents and nodes are removed from the vector index
- Retriever is rebuilt to reflect changes
- System maintains consistency between storage and index

## Technical Details

### Architecture
- **FastAPI**: Modern, fast web framework
- **LlamaIndex**: Document processing and vector storage
- **Gemma Model**: Large language model for text generation
- **Persian Embeddings**: Specialized embeddings for Persian text
- **Sentence Splitting**: Intelligent document chunking

### Data Flow
1. Document upload → Chunking → Embedding → Vector storage
2. Query → Retrieval → Context assembly → LLM generation
3. Deletion → Index cleanup → Filesystem cleanup → Retriever rebuild

### Error Handling
- Duplicate file detection (by content hash and filename)
- Graceful handling of missing files
- Comprehensive error messages and status codes
- Automatic cleanup on failures

## Troubleshooting

### Common Issues

1. **File not found errors:**
   - Ensure the file exists in the `./data` directory
   - Check file permissions

2. **Index inconsistencies:**
   - Restart the application to rebuild the index
   - Check for corrupted metadata

3. **Memory issues:**
   - Reduce chunk size in settings
   - Limit the number of concurrent operations

### Performance Tips

- Use appropriate chunk sizes for your documents
- Monitor memory usage during large file operations
- Consider batch processing for multiple files
- Regular cleanup of unused files

## Contributing

When adding new features:
1. Update the test suite
2. Add proper error handling
3. Document new endpoints
4. Test with various file types and sizes

## License

This project is for educational and research purposes.

