import os
import sys
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from openai import OpenAI
from datetime import datetime

# from src.exception.exception import NewsArticleException
# from src.logger.logger import logging
import pymongo
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
opneai_client = OpenAI(api_key=OPENAI_API_KEY)

MONGO_DB_URL = os.getenv("MONGO_URI")


class ArticleIngestion:
    def __init__(self, sources: List[Dict]):
        try:
            self.sources = sources
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.mongo_db = self.mongo_client["news_data"]
            self.mongo_collection = self.mongo_db["articles"]
        except Exception as e:
            raise Exception(f"Error initializing MongoDB: {e}") from e

    def scrape_articles(self, base_url: str, target_date: str) -> List[Dict]:
        try:
            response = requests.get(base_url)
            if response.status_code != 200:
                # logging.warning(f"Failed to fetch page: {base_url}")
                return []

            soup = BeautifulSoup(response.content, "html.parser")
            articles = []

            for h3 in soup.find_all("h3"):
                a_tag = h3.find("a", href=True)
                if a_tag:
                    title = h3.get_text(strip=True)
                    link = a_tag["href"]
                    if link.startswith("/"):
                        link = base_url.rstrip("/") + link
                    if target_date in link:
                        articles.append({"title": title, "link": link})
            return articles

        except Exception as e:
            raise Exception(f"Error scraping articles from {base_url}: {e}") from e

    def extract_article_content(self, article: Dict) -> Dict:
        try:
            url = article["link"]
            response = requests.get(url)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, "html.parser")
            title_element = soup.select_one("h1")
            title = (
                title_element.get_text(strip=True)
                if title_element
                else article["title"]
            )

            paragraphs = soup.find_all("div", class_="paragraph") or soup.find_all("p")
            content = (
                " ".join(
                    p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
                )
                or "No content found."
            )

            return {"title": title, "content": content, "url": url}

        except Exception as e:
            raise Exception(f"Error extracting content from article: {e}") from e

    def clean_article(self, text: str, source_url: str) -> str:
        try:
            if "abc.net.au" in source_url:
                match = re.search(r"Topic:(\w+)\s+(.*?)(?=Topic:|$)", text, re.DOTALL)
                if match:
                    topic = match.group(1).strip()
                    content = match.group(2).strip()
                    return f"{topic} {content}"
                return ""

            elif "smh.com.au" in source_url:
                text = text.replace(
                    "We’re sorry, this feature is currently unavailable. We’re working to restore it. Please try again later. Add articles to your saved list and come back to them any time.",
                    "",
                ).replace("Reuters Copyright ©2025", "")
                return " ".join(text.split()).strip()

            elif "nine.com.au" in source_url:
                text = text.replace("Nine’s Wide World of Sports", "").replace(
                    "©2025Nine Entertainment Co", ""
                )
                return " ".join(text.split()).strip()

            return text

        except Exception as e:
            raise Exception(f"Error cleaning article from {source_url}: {e}") from e

    def get_openai_embedding(
        self, text: str, model: str = "text-embedding-ada-002"
    ) -> List[float]:
        try:
            response = opneai_client.embeddings.create(input=text[:8191], model=model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    def initiate_ingestion(self):
        documents_for_vector_store = []
        try:
            # logging.info("Starting article ingestion and MongoDB storage...")
            id_counter = 0

            for source in self.sources:
                # logging.info(f"Scraping from {source['url']}")
                articles = self.scrape_articles(source["url"], source["date"])

                for article in articles:
                    result = self.extract_article_content(article)
                    if result:
                        cleaned_text = self.clean_article(
                            result["content"], result["url"]
                        )
                        # Generate embedding from title + content
                        combined_input = f"{result['title']} {cleaned_text}"
                        embedding = self.get_openai_embedding(combined_input)
                        if not embedding:
                            continue

                        # document = {
                        #     "id": id_counter,
                        #     "source": result["url"].split("/")[2],
                        #     "category": "sports",
                        #     "title": result["title"],
                        #     "content": cleaned_text,
                        #     "url": result["url"],
                        # }
                        document = {
                            "title": result["title"],
                            "text": cleaned_text,
                            "url": result["url"],
                            "source": result["url"].split("/")[2],
                            "category": "sports",
                            "published_date": None,
                            "embedding": embedding,
                            "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        documents_for_vector_store.append(document)

                        self.mongo_collection.insert_one(document)
                        # logging.info(f"Inserted article ID: {id_counter}")
                        id_counter += 1
            return documents_for_vector_store
            # logging.info("✅ Article ingestion completed and stored in MongoDB.")

        except Exception as e:
            raise Exception(f"Error during ingestion pipeline: {e}") from e
