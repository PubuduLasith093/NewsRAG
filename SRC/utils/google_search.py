from googlesearch import search
from datetime import datetime, timedelta
from typing import List, Dict
import time


def get_google_news_from_aus_last_24_hours(
    query: str = "", num_results: int = 100
) -> List[Dict]:
    """
    Fetches news links from Australian (.au) domains published within the last 24 hours using Google Search.

    Args:
        query (str): Optional search query keywords.
        num_results (int): Number of news results to retrieve (up to 100).

    Returns:
        List[Dict]: List of result dictionaries containing title and link.
    """
    # Get yesterday's date in YYYY-MM-DD format
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Google advanced search query to filter:
    # - Australian websites (site:.au)
    # - Pages indexed after yesterday (past 24 hours)
    full_query = f"{query} site:.au after:{yesterday}"

    print(f"🔍 Searching Google with query: {full_query}")

    try:
        results = []
        for url in search(full_query, num_results=num_results):
            results.append({"url": url})
            time.sleep(0.5)  # Slight delay to avoid rate-limiting
        return results
    except Exception as e:
        raise RuntimeError(f"Google search failed: {e}")


def example_usage():
    print("🗞️ Google News from Australia (last 24 hours):\n")
    articles = get_google_news_from_aus_last_24_hours("business", 100)
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article['url']}")


if __name__ == "__main__":
    example_usage()
