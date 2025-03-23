from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
# MONGO_URI = os.getenv("MONGO_URI")

# 🔑 High-priority keywords
HIGHLIGHT_KEYWORDS = ["breaking news", "exclusive", "just in", "confirmed", "revealed"]


def is_highlight_keyword_present(text: str):
    return any(kw in text.lower() for kw in HIGHLIGHT_KEYWORDS)


def get_highlights(db, category: str, top_k=5, similarity_threshold=0.9):
    collection = db["articles"]

    # Step 1: Get articles in this category
    articles = list(collection.find({"predicted_topic": category}))
    if not articles:
        return []

    print(f"📚 Found {len(articles)} articles in category '{category}'")

    # Step 2: Compute cosine similarity matrix between embeddings
    embeddings = np.array([article["embedding"] for article in articles])
    sim_matrix = cosine_similarity(embeddings)

    # Step 3: Frequency scoring - count similar articles (above threshold)
    freq_scores = (sim_matrix > similarity_threshold).sum(axis=1)

    # Step 4: Keyword priority
    for i, article in enumerate(articles):
        has_keywords = is_highlight_keyword_present(
            article.get("title", "") + article.get("text", "")
        )
        freq_scores[i] += 5 if has_keywords else 0  # Add a boost if keyword found

    # Step 5: Rank and return top_k highlights
    top_indices = np.argsort(freq_scores)[-top_k:][::-1]
    top_articles = [articles[i] for i in top_indices]

    return top_articles


def update_featured_highlights(
    mongo_client: MongoClient, category: str, top_k: int = 5
):
    """
    Update articles in MongoDB for a given category:
      - Mark all articles as not featured.
      - Then mark the top highlights as featured so they can be shown in the UI.
    """
    db = mongo_client["news_data"]
    collection = db["articles"]

    # Mark all articles in this category as not featured
    collection.update_many(
        {"predicted_topic": category}, {"$set": {"is_featured": False}}
    )

    # Get top highlight articles for the category
    highlights = get_highlights(db, category, top_k=top_k)
    highlight_ids = [doc["_id"] for doc in highlights]

    # Mark highlighted articles as featured with a timestamp
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for doc_id in highlight_ids:
        collection.update_one(
            {"_id": doc_id}, {"$set": {"is_featured": True, "featured_at": now_str}}
        )

    print(
        f"Updated {len(highlight_ids)} articles as featured in category '{category}'."
    )


def run_highlights_for_all_categories(mongo_client, categories: list) -> None:
    """
    Loop over all given categories and update featured highlights.
    """
    for category in categories:
        print(f"\nUpdating featured highlights for category: {category}")
        update_featured_highlights(mongo_client, category)


def run_highlight(mongo_client):
    categories = [
        "sports",
        "finance",
        "lifestyle",
        "music",
        "technology",
        "politics",
        "health",
        "education",
    ]
    run_highlights_for_all_categories(mongo_client, categories)
