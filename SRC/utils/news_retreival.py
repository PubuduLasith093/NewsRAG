import os
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


def fetch_latest_news(
    api_key: str,
    countries: List[str] = None,
    language: str = "en",
    page: int = 1,
    page_size: int = 10,
    timeframe: int = 24,
) -> List[Dict[str, Any]]:
    """
    Fetch the latest news articles from the NewsData.io API for specified countries.

    Args:
        api_key (str): Your NewsData.io API key.
        countries (List[str], optional): List of country codes (e.g. ["au", "us"]).
                                         Defaults to None (which uses ["au", "us"]).
        language (str, optional): News language code (e.g. 'en' for English).
                                  Defaults to 'en'.
        page (int, optional): Page number for pagination. Defaults to 1.
        page_size (int, optional): Number of articles per page, if supported by the API.
                                   Defaults to 10.
        timeframe (int, optional): Search the news articles for a specific timeframe (Minutes and Hours).
                                   For hours, you can set a timeframe of 1 to 48, and for minutes, you
                                   can define a timeframe of 1m to 2880m.

    Returns:
        List[Dict[str, Any]]: A list of news articles (each article is a dict containing
                              fields such as title, description, url, etc.).

    Raises:
        ValueError: If the request returns a non-200 status code or invalid data.
    """
    if countries is None:
        countries = ["au"]  # default to Australia

    # Construct the base URL using your parameters.
    # Refer to https://newsdata.io/docs for additional query parameters if needed.
    base_url = "https://newsdata.io/api/1/latest"

    # Convert the list of countries to a comma-separated string.
    country_param = ",".join(countries)

    # Prepare query parameters
    params = {
        "apikey": api_key,
        "country": country_param,
        # "language": language,
        # "page": page,
        # Note: Some fields like 'page_size' may not be directly supported by this API,
        # but we'll keep it here if you need to adapt for a different endpoint or if
        # the API eventually supports it.
        # "page_size": page_size,
        # "timeframe": timeframe,
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
    except requests.exceptions.RequestException as e:
        # This block captures network errors or timeouts
        raise ValueError(f"Network-related error occurred: {e}") from e

    if response.status_code != 200:
        # Raise an error if the API call was not successful
        raise ValueError(
            f"Request to NewsData.io failed with status code {response.status_code}. "
            f"Response text: {response.text}"
        )

    data = response.json()

    # 'results' typically holds the array of articles in NewsData.io's response
    # (See the API documentation for exact field names.)
    articles = data.get("results", [])
    if not isinstance(articles, list):
        raise ValueError("Invalid response format: 'results' is not a list.")

    return articles


def example_usage():
    """
    Example usage of the fetch_latest_news function.
    Replace 'YOUR_API_KEY' with your actual key before running.
    """
    NEWSIO_API_KEY = os.getenv("NEWSDATA_IO_API_KEY")
    try:
        news_articles = fetch_latest_news(
            api_key=NEWSIO_API_KEY,
        )

        print(f"Retrieved {len(news_articles)} articles.")
        print(news_articles[0])
        for article in news_articles:
            # Print a small sample of each article's data
            print("-" * 80)
            print("Title:       ", article.get("title"))
            print("Description: ", article.get("description"))
            print("URL:         ", article.get("link"))
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_usage()
