import os
from pymongo import MongoClient
from transformers import pipeline
from dotenv import load_dotenv
from tqdm import tqdm
from SRC.utils.fetch_articles import fetch_articles_with_embeddings

# Load env
load_dotenv()
# MONGO_URI = os.getenv("MONGO_URI")

# Define your topic labels (can be adjusted)
TOPIC_LABELS = [
    "sports",
    "finance",
    "lifestyle",
    "music",
    "technology",
    "politics",
    "health",
    "education",
]

# Load classifier (this will download the model if not cached)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def classify_article(article_text: str) -> str:
    """
    Use zero-shot classification to label the article.
    Returns the highest-scoring label.
    """
    result = classifier(article_text, TOPIC_LABELS, multi_label=False)
    return result["labels"][0]  # Best label


def update_article_topic(db, collection_name: str, article_id, topic: str):
    """
    Update a single article with its predicted topic.
    """
    collection = db[collection_name]
    collection.update_one({"_id": article_id}, {"$set": {"predicted_topic": topic}})


def zeroshot_topic_clustering(mongo_client):
    # Connect to MongoDB
    db = mongo_client["news_data"]
    collection_name = "articles"

    # Fetch articles that don't have a predicted_topic yet
    articles = fetch_articles_with_embeddings(db, collection_name)
    print(f"Found {len(articles)} articles to classify.")

    for article in tqdm(articles):
        text = f"{article.get('title', '')}\n{article.get('text', '')}"[
            :2048
        ]  # truncate if too long
        predicted_topic = classify_article(text)
        update_article_topic(
            db,
            collection_name,
            article["_id"],
            predicted_topic,
        )
        print(f"✅ [{predicted_topic}] - {article.get('title', '')[:80]}")


# if __name__ == "__main__":
#     zeroshot_topic_clustering()
