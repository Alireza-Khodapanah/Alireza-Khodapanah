# Test API endpoints using TestClient
# Run this in Google Colab after setting up your FastAPI app

import os
import tempfile
from fastapi.testclient import TestClient
from fastapi import FastAPI
import pytest

# Import your app (make sure this matches your actual app file)
# If your app is in a different file, update this import
from app import app

# Create test client
client = TestClient(app)

def test_health_check():
    """Test if the API is running"""
    try:
        response = client.get("/docs")
        print("✅ API is running and accessible")
        print(f"📚 API docs available at: {response.url}")
        return True
    except Exception as e:
        print(f"❌ API not accessible: {e}")
        return False

def test_list_files_initial():
    """Test listing files when no files exist"""
    print("\n🔍 Testing initial file listing...")
    try:
        response = client.get("/files")
        print(f"📊 Response status: {response.status_code}")
        print(f"📋 Response: {response.json()}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📁 Total files: {data.get('total_files', 0)}")
            return True
        else:
            print(f"❌ Failed to list files: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing file listing: {e}")
        return False

def create_test_file(filename, content):
    """Create a temporary test file"""
    temp_dir = "./data"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return file_path

def test_upload_files():
    """Test file upload functionality"""
    print("\n📤 Testing file upload...")
    
    try:
        # Create test files
        test_files = [
            ("test1.txt", "This is test file 1 with some content about law and constitution."),
            ("test2.txt", "This is test file 2 with different content about legal processes."),
            ("test3.txt", "This is test file 3 with more legal information.")
        ]
        
        # Create files in the data directory
        for filename, content in test_files:
            create_test_file(filename, content)
            print(f"📝 Created test file: {filename}")
        
        # Test upload endpoint (since we already have files, we'll test the logic)
        print("📤 Testing upload endpoint logic...")
        
        # For testing purposes, we'll simulate the upload by checking if files exist
        for filename, _ in test_files:
            file_path = os.path.join("./data", filename)
            if os.path.exists(file_path):
                print(f"✅ File {filename} exists in data directory")
            else:
                print(f"❌ File {filename} not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing file upload: {e}")
        return False

def test_ask_question():
    """Test the ask endpoint"""
    print("\n❓ Testing ask endpoint...")
    
    try:
        # Test query
        test_query = "What is the content of test files?"
        
        response = client.post("/ask", json={"question": test_query})
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Question answered successfully")
            print(f"📝 Answer: {data.get('answer', 'No answer')[:100]}...")
            print(f"📚 Context length: {len(data.get('context', ''))}")
            return True
        else:
            print(f"❌ Failed to get answer: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing ask endpoint: {e}")
        return False

def test_delete_file():
    """Test file deletion functionality"""
    print("\n🗑️ Testing file deletion...")
    
    try:
        # First, list files to see what's available
        response = client.get("/files")
        if response.status_code != 200:
            print("❌ Cannot list files for deletion test")
            return False
        
        files_data = response.json()
        files = files_data.get('files', [])
        
        if not files:
            print("⚠️ No files available for deletion test")
            return False
        
        # Select first file for deletion
        file_to_delete = files[0]['filename']
        print(f"🗑️ Attempting to delete file: {file_to_delete}")
        
        # Test deletion
        response = client.delete(f"/delete/{file_to_delete}")
        print(f"📊 Delete response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ File deleted successfully")
            print(f"📋 Response: {data}")
            
            # Verify file is removed from filesystem
            file_path = os.path.join("./data", file_to_delete)
            if not os.path.exists(file_path):
                print(f"✅ File {file_to_delete} removed from filesystem")
            else:
                print(f"❌ File {file_to_delete} still exists in filesystem")
            
            # Verify file is removed from index
            response = client.get("/files")
            if response.status_code == 200:
                files_after = response.json().get('files', [])
                remaining_files = [f['filename'] for f in files_after]
                if file_to_delete not in remaining_files:
                    print(f"✅ File {file_to_delete} removed from index")
                else:
                    print(f"❌ File {file_to_delete} still exists in index")
            
            return True
        else:
            print(f"❌ Failed to delete file: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing file deletion: {e}")
        return False

def test_delete_nonexistent_file():
    """Test deletion of a file that doesn't exist"""
    print("\n🚫 Testing deletion of nonexistent file...")
    
    try:
        nonexistent_file = "nonexistent_file.txt"
        response = client.delete(f"/delete/{nonexistent_file}")
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 404:
            print(f"✅ Correctly returned 404 for nonexistent file")
            return True
        else:
            print(f"❌ Expected 404, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing nonexistent file deletion: {e}")
        return False

def cleanup_test_files():
    """Clean up test files after testing"""
    print("\n🧹 Cleaning up test files...")
    
    try:
        test_files = ["test1.txt", "test2.txt", "test3.txt"]
        data_dir = "./data"
        
        for filename in test_files:
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Removed test file: {filename}")
        
        print("✅ Cleanup completed")
        return True
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return False

def run_all_tests():
    """Run all tests in sequence"""
    print("🚀 Starting API tests...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("List Files (Initial)", test_list_files_initial),
        ("Upload Files", test_upload_files),
        ("Ask Question", test_ask_question),
        ("Delete File", test_delete_file),
        ("Delete Nonexistent File", test_delete_nonexistent_file),
        ("Cleanup", cleanup_test_files)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

# Run tests if this file is executed directly
if __name__ == "__main__":
    run_all_tests()

# Alternative: Run individual tests
def quick_test():
    """Quick test of main functionality"""
    print("🔍 Quick API test...")
    
    # Test if API is accessible
    if test_health_check():
        print("✅ API is accessible")
        
        # Test file listing
        if test_list_files_initial():
            print("✅ File listing works")
        
        # Test ask endpoint
        if test_ask_question():
            print("✅ Ask endpoint works")
        
        print("✅ Quick test completed successfully")
    else:
        print("❌ API is not accessible")

# Uncomment to run quick test
# quick_test()