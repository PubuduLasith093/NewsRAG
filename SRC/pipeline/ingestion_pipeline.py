import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from SRC.utils.news_sites_data_pipeline import ArticleIngestion
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from openai import OpenAI
from dotenv import load_dotenv
from googlesearch import search
from newspaper import Article

from sentence_transformers import (
    SentenceTransformer,
)  # (if you ever want to switch models)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
MONGO_URI = os.getenv("MONGO_URI")


def format_scraper_date(scraper_date_str: str) -> str:
    """
    Convert date string from 'YYYYMMDD' to 'YYYY-MM-DD'
    """
    return datetime.strptime(scraper_date_str, "%Y%m%d").strftime("%Y-%m-%d")


# Use OpenAI for embeddings (text-embedding-ada-002 produces 1536-dim vectors)
def get_openai_embedding(
    text: str, model: str = "text-embedding-ada-002"
) -> List[float]:
    """
    Generate an embedding using OpenAI's API.
    Truncates input to avoid token limits.
    """
    try:
        response = client.embeddings.create(
            input=text[:8191], model=model
        )  # Ensure within token limits)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error in embedding generation: {e}")
        return []


def get_yesterday_date_str() -> str:
    """Return yesterday's date in YYYY-MM-DD format."""
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def google_news_urls(category: str, max_results: int = 25) -> List[str]:
    """
    Uses Google Search to fetch news URLs from Australian domains in the past 24 hours.
    It builds a query with the specified category and a time filter.
    """
    # Build query: category + restrict to .au domains + only news after yesterday
    query = f"{category} site:.au after:{get_yesterday_date_str()}"
    print(f"🔍 Searching for: {query}")
    try:
        return list(search(query, num_results=max_results))
    except Exception as e:
        print(f"Error during Google Search: {e}")
        return []


def extract_article_metadata(url: str) -> Dict:
    """
    Extracts article metadata and full text from a given URL using Newspaper3k.
    Returns a dictionary with keys: title, url, authors, published_date, text, top_image, source.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            "title": article.title,
            "url": url,
            "authors": article.authors,
            "published_date": (
                article.publish_date.strftime("%Y-%m-%d %H:%M:%S")
                if article.publish_date
                else None
            ),
            "text": article.text,
            "top_image": article.top_image,
            "source": article.source_url,
        }
    except Exception as e:
        print(f"⚠️ Failed to extract article from {url}: {e}")
        return {}


def ingest_category(
    collection,
    category: str,
    limit: int = 10,
):
    """
    Ingests news articles for a specified category.
    For each article URL, it extracts metadata, computes an embedding, and saves it to MongoDB.
    """
    urls = google_news_urls(
        category, max_results=limit * 2
    )  # extra results to account for failures
    stored_count = 0
    to_vectorize = []

    for url in urls:
        if stored_count >= limit:
            break

        # Avoid duplicates by URL
        if collection.find_one({"url": url}):
            continue

        article_data = extract_article_metadata(url)
        if not article_data.get("text"):
            continue

        # Combine title and text for embedding to capture context
        embed_input = f"{article_data['title']} {article_data['text']}"
        embedding = get_openai_embedding(embed_input)
        if not embedding:
            continue

        # Add additional fields
        article_data["category"] = category
        article_data["embedding"] = embedding  # 1536-dim vector from OpenAI
        article_data["ingested_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        collection.insert_one(article_data)
        to_vectorize.append(
            {
                "title": article_data["title"],
                "text": article_data["text"],
                "url": article_data["url"],
                "source": article_data.get("source"),
                "published_date": article_data.get("published_date"),
                "category": category,
                "embedding": embedding,
            }
        )

        print(f"✅ Stored article: {article_data['title']}")
        stored_count += 1
        time.sleep(0.5)  # polite delay
    upsert_to_vector_store(to_vectorize)


# def upsert_to_vector_store(documents: List[Dict]):
#     if not documents:
#         return
#     embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
#     vector_store = MongoDBAtlasVectorSearch.from_connection_string(
#         connection_string=MONGO_URI,
#         namespace="news_data.articles",
#         embedding=embeddings,
#         index_name="news_index",
#     )
#     vector_store.add_documents(documents)
#     print("✅ Vector store updated")

from langchain.schema import Document


def upsert_to_vector_store(documents: List[Dict]):
    if not documents:
        return

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=MONGO_URI,
        namespace="news_data.articles",
        embedding=embeddings,
        index_name="news_index",
    )

    docs = []
    for doc in documents:
        content = f"{doc['title']} {doc['text']}"
        metadata = {
            "title": doc["title"],
            "text": doc["text"],
            "url": doc["url"],
            "source": doc.get("source", ""),
            "published_date": doc.get("published_date", ""),
            "category": doc.get("category", ""),
        }
        docs.append(Document(page_content=content, metadata=metadata))

    vector_store.add_documents(docs)
    print("✅ Vector store updated")


def run_ingestion_pipeline(mongo_client) -> None:
    """
    Main function that runs the ingestion pipeline for multiple categories.
    This function can be imported and called from main.py.
    """
    # mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["news_data"]
    collection = db["articles"]
    categories = ["sports", "lifestyle", "music", "finance"]
    for category in categories:
        logger.info(f"📥 Ingesting category: {category}")
        ingest_category(
            collection,
            category,
            limit=5,
        )

    logger.info("⚙️ Running structured website scraping...")
    today_str = datetime.now().strftime("%Y%m%d")  # e.g. '20250322'
    sources = [
        {"url": "https://www.abc.net.au/news/sport", "date": today_str},
        {"url": "https://www.smh.com.au/sport", "date": today_str},
        {"url": "https://www.nine.com.au/sport", "date": today_str},
    ]

    structured_ingestion = ArticleIngestion(sources=sources)
    documents_for_vector_store = structured_ingestion.initiate_ingestion()
    upsert_to_vector_store(documents_for_vector_store)


# if __name__ == "__main__":
#     run_ingestion_pipeline()
