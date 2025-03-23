from datetime import datetime, timedelta
from newspaper import Article
from googlesearch import search
from typing import List, Dict
import time


def get_yesterday_date_str() -> str:
    """Returns yesterday's date in YYYY-MM-DD format (used for Google time filtering)."""
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def google_news_urls_by_category(category: str, max_results: int = 25) -> List[str]:
    """
    Uses Google Search to get news URLs related to a category from .au domains (last 24 hours).

    Args:
        category (str): Category of news (e.g., 'sports', 'finance').
        max_results (int): Number of URLs to retrieve.

    Returns:
        List[str]: List of article URLs.
    """
    yesterday = get_yesterday_date_str()
    query = f"{category} site:.au after:{yesterday}"
    print(f"🔍 Searching Google: {query}")
    results = search(query, num_results=max_results)
    return list(results)


def extract_article_metadata(url: str) -> Dict:
    """
    Extracts full article content and metadata using newspaper3k.

    Args:
        url (str): URL of the article.

    Returns:
        Dict: Dictionary containing article metadata and text.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        # Optional NLP: article.nlp()

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
        print(f"⚠️ Failed to extract from {url}: {e}")
        return {}


def fetch_news_by_category(category: str, limit: int = 10) -> List[Dict]:
    """
    Main pipeline: Fetches full articles for a given news category.

    Args:
        category (str): News category (e.g., 'finance', 'music').
        limit (int): Max number of news articles to fetch.

    Returns:
        List[Dict]: List of full article metadata dicts.
    """
    urls = google_news_urls_by_category(
        category, max_results=limit * 2
    )  # extra to account for failures
    articles = []

    for url in urls:
        if len(articles) >= limit:
            break
        article_data = extract_article_metadata(url)
        if article_data.get("text"):
            articles.append(article_data)
            print(f"✅ Fetched: {article_data['title']}")
        time.sleep(0.5)  # polite delay to avoid being blocked

    return articles


def run_news_pipeline():
    """
    Fetches news articles for multiple Australian categories in the last 24 hours.
    """
    categories = ["sports", "lifestyle", "music", "finance"]
    all_news = {}

    for cat in categories:
        print(f"\n📢 Collecting news for category: {cat.capitalize()}")
        all_news[cat] = fetch_news_by_category(
            cat, limit=5
        )  # fetch 5 articles per category

    return all_news


if __name__ == "__main__":
    news_data = run_news_pipeline()

    for category, articles in news_data.items():
        print(f"\n\n📚 {category.upper()} ARTICLES")
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. 📰 {article['title']}")
            print(f"🔗 URL: {article['url']}")
            print(f"👤 Authors: {article['authors']}")
            print(f"📅 Published: {article['published_date']}")
            print(f"📝 Content Preview: {article['text'][:200]}...")
