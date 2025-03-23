from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

# Replace with your DB and collection
db = client["news_data"]
collection = db["articles"]

# Create vector index manually (if not already created)
try:
    db.command(
        {
            "createSearchIndex": "news_index",
            "collection": "articles",
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "embedding": {
                            "type": "vector",
                            "dimensions": 1536,
                            "similarity": "cosine",
                        }
                    },
                }
            },
        }
    )
    print("✅ Vector index 'news_index' created.")
except Exception as e:
    print(f"⚠️ Index may already exist or failed: {e}")
