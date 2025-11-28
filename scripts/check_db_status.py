import chromadb
from chromadb.config import Settings
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_db():
    print("Checking ChromaDB status...")
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = client.list_collections()
        print(f"Found {len(collections)} collections:")
        
        for col in collections:
            count = col.count()
            print(f"- Name: {col.name}, Count: {count}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db()



