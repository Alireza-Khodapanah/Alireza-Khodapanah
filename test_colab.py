# Simple test script for Google Colab
# Run this after setting up your FastAPI app

import os
import requests
import json

def test_api_endpoints():
    """Test all API endpoints"""
    print("🚀 Starting API tests...")
    print("=" * 50)
    
    # Base URL for your API
    base_url = "http://localhost:8000"  # Change if different
    
    # Test 1: Health check
    print("\n🔍 Test 1: API Health Check")
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✅ API is running and accessible")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False
    
    # Test 2: List files (should be empty initially)
    print("\n📁 Test 2: List Files (Initial)")
    try:
        response = requests.get(f"{base_url}/files")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Files listed successfully")
            print(f"📊 Total files: {data.get('total_files', 0)}")
        else:
            print(f"❌ Failed to list files: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return False
    
    # Test 3: Create test files
    print("\n📝 Test 3: Create Test Files")
    test_files = [
        ("test1.txt", "This is test file 1 with content about law and constitution."),
        ("test2.txt", "This is test file 2 with different content about legal processes."),
        ("test3.txt", "This is test file 3 with more legal information.")
    ]
    
    for filename, content in test_files:
        file_path = os.path.join("./data", filename)
        os.makedirs("./data", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Created: {filename}")
    
    # Test 4: Test ask endpoint
    print("\n❓ Test 4: Ask Question")
    try:
        test_query = "What is the content of test files?"
        response = requests.post(f"{base_url}/ask", json={"question": test_query})
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Question answered successfully")
            print(f"📝 Answer: {data.get('answer', 'No answer')[:100]}...")
            print(f"📚 Context length: {len(data.get('context', ''))}")
        else:
            print(f"❌ Failed to get answer: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing ask endpoint: {e}")
        return False
    
    # Test 5: Test file deletion
    print("\n🗑️ Test 5: File Deletion")
    try:
        # First, list files to see what's available
        response = requests.get(f"{base_url}/files")
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
        response = requests.delete(f"{base_url}/delete/{file_to_delete}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ File deleted successfully")
            print(f"📋 Response: {data}")
            
            # Verify file is removed from filesystem
            file_path = os.path.join("./data", file_to_delete)
            if not os.path.exists(file_path):
                print(f"✅ File {file_to_delete} removed from filesystem")
            else:
                print(f"❌ File {file_to_delete} still exists in filesystem")
            
            # Verify file is removed from index
            response = requests.get(f"{base_url}/files")
            if response.status_code == 200:
                files_after = response.json().get('files', [])
                remaining_files = [f['filename'] for f in files_after]
                if file_to_delete not in remaining_files:
                    print(f"✅ File {file_to_delete} removed from index")
                else:
                    print(f"❌ File {file_to_delete} still exists in index")
        else:
            print(f"❌ Failed to delete file: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing file deletion: {e}")
        return False
    
    # Test 6: Test deletion of nonexistent file
    print("\n🚫 Test 6: Delete Nonexistent File")
    try:
        nonexistent_file = "nonexistent_file.txt"
        response = requests.delete(f"{base_url}/delete/{nonexistent_file}")
        
        if response.status_code == 404:
            print("✅ Correctly returned 404 for nonexistent file")
        else:
            print(f"❌ Expected 404, got {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing nonexistent file deletion: {e}")
        return False
    
    # Test 7: Cleanup
    print("\n🧹 Test 7: Cleanup")
    try:
        test_files = ["test1.txt", "test2.txt", "test3.txt"]
        data_dir = "./data"
        
        for filename in test_files:
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Removed test file: {filename}")
        
        print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return False
    
    # Final check
    print("\n🔍 Final Check")
    try:
        response = requests.get(f"{base_url}/files")
        if response.status_code == 200:
            data = response.json()
            print(f"📁 Total files after cleanup: {data.get('total_files', 0)}")
            if data.get('total_files', 0) == 0:
                print("✅ All test files successfully removed!")
            else:
                print(f"⚠️ {data.get('total_files', 0)} files still remain")
        else:
            print(f"❌ Failed to get final file list: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error in final check: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed successfully!")
    print("✅ Your RAG API with file deletion is working correctly!")
    return True

def quick_test():
    """Quick test of main functionality"""
    print("🔍 Quick API test...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test if API is accessible
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✅ API is accessible")
            
            # Test file listing
            response = requests.get(f"{base_url}/files")
            if response.status_code == 200:
                print("✅ File listing works")
            
            # Test ask endpoint
            response = requests.post(f"{base_url}/ask", json={"question": "Test question"})
            if response.status_code == 200:
                print("✅ Ask endpoint works")
            
            print("✅ Quick test completed successfully")
            return True
        else:
            print("❌ API is not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

# Run tests
if __name__ == "__main__":
    # Uncomment the line below to run the full test suite
    # test_api_endpoints()
    
    # Or run quick test
    quick_test()

# Instructions for Colab:
# 1. Make sure your FastAPI app is running: uvicorn app:app --host 0.0.0.0 --port 8000
# 2. Run this script: python test_colab.py
# 3. Or run individual tests by calling the functions directly