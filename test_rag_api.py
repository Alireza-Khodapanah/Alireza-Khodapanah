# Test code for RAG API with FastAPI TestClient
# Run this in Google Colab after installing the required packages

import asyncio
import io
import os
import tempfile
from fastapi.testclient import TestClient
from rag_api import app
import json

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

# Additional utility functions for interactive testing

def interactive_upload_test():
    """Interactive test for uploading files."""
    print("📁 Interactive Upload Test")
    print("This will create and upload test files...")
    
    # Create multiple test files with Persian content
    test_files_content = {
        "document1.txt": """
        قانون اساسی جمهوری اسلامی ایران
        
        فصل اول: اصول کلی
        
        اصل یکم: حکومت ایران، جمهوری اسلامی است که ملت ایران بر اساس ایمان دیرینه‌اش به حاکمیت حق و عدالت قرآن، در پانزدهم خرداد ماه یکهزار و سیصد و پنجاه و هشت هجری شمسی معادل بیست و هشتم ذی‌القعده یکهزار و سیصد و نود و نه هجری قمری با رأی مثبت نود و هشت و هفت دهم درصد آراء شرکت‌کنندگان در همه‌پرسی تصویب نمود.
        """,
        "document2.txt": """
        اصول حقوق شهروندی
        
        هر شهروند دارای حقوق و تکالیف مشخصی است که در قانون اساسی تعریف شده است.
        حق آزادی بیان، حق تجمع، و حق دسترسی به اطلاعات از جمله حقوق اساسی شهروندان محسوب می‌شود.
        """,
        "document3.txt": """
        فرآیند قانون‌گذاری
        
        قوانین در مجلس شورای اسلامی تصویب می‌شود.
        پس از تصویب در مجلس، قوانین برای بررسی نهایی به شورای نگهبان ارسال می‌شود.
        در صورت تأیید شورای نگهبان، قانون برای اجرا ابلاغ می‌شود.
        """
    }
    
    files = []
    for filename, content in test_files_content.items():
        files.append(("files", (filename, create_test_file(filename, content), "text/plain")))
    
    response = client.post("/upload", files=files)
    print(f"Upload Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return list(test_files_content.keys())

def interactive_query_test():
    """Interactive test for querying."""
    print("❓ Interactive Query Test")
    
    queries = [
        "قانون اساسی چگونه تصویب شد؟",
        "حقوق شهروندی چیست؟",
        "فرآیند قانون‌گذاری چگونه است؟",
        "شورای نگهبان چه نقشی دارد؟"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        response = client.post("/ask", json={"question": query})
        if response.status_code == 200:
            data = response.json()
            print(f"Answer: {data['answer'][:200]}...")  # Show first 200 chars
        else:
            print(f"Error: {response.status_code} - {response.text}")

def interactive_delete_test(filenames):
    """Interactive test for deleting files."""
    print("🗑️ Interactive Delete Test")
    
    for filename in filenames:
        print(f"\nDeleting: {filename}")
        response = client.delete("/delete", json={"filename": filename})
        print(f"Delete Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# Main execution
if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Run all automated tests")
    print("2. Run interactive tests")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_all_tests()
    elif choice == "2":
        print("\n🔧 Running Interactive Tests...")
        
        # Upload test files
        uploaded_files = interactive_upload_test()
        
        # List files
        files_response = client.get("/files")
        print(f"\n📋 Current files: {json.dumps(files_response.json(), indent=2, ensure_ascii=False)}")
        
        # Test queries
        interactive_query_test()
        
        # Test deletion
        interactive_delete_test(uploaded_files)
        
        # Final file list
        final_files = client.get("/files")
        print(f"\n📋 Final files: {json.dumps(final_files.json(), indent=2, ensure_ascii=False)}")
        
    else:
        print("Invalid choice. Running automated tests...")
        run_all_tests()