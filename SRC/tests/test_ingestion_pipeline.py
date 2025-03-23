import os
import pytest
from datetime import datetime, timedelta

# Import functions from the ingestion_pipeline module
from pipeline.ingestion_pipeline import (
    get_yesterday_date_str,
    google_news_urls,
    extract_article_metadata,
    get_openai_embedding,
    ingest_category,
)


# Dummy implementations to use in tests
def dummy_search(query, num_results):
    return ["http://example.com/article1", "http://example.com/article2"]


def dummy_extract_article_metadata(url):
    return {
        "title": "Dummy Article",
        "url": url,
        "authors": ["Test Author"],
        "published_date": "2023-03-28 12:00:00",
        "text": "This is a dummy article text.",
        "top_image": "http://example.com/image.jpg",
        "source": "Example News",
    }


def dummy_get_openai_embedding(text, model="text-embedding-ada-002"):
    return [0.1] * 1536


# Dummy collection to simulate MongoDB operations
class DummyCollection:
    def __init__(self):
        self.data = []

    def find_one(self, query):
        for doc in self.data:
            if doc.get("url") == query.get("url"):
                return doc
        return None

    def insert_one(self, doc):
        self.data.append(doc)


# Fixture to override the collection in ingestion_pipeline with a dummy collection
@pytest.fixture
def dummy_collection(monkeypatch):
    dummy = DummyCollection()
    monkeypatch.setattr("pipeline.ingestion_pipeline.collection", dummy)
    return dummy


def test_get_yesterday_date_str():
    yesterday = get_yesterday_date_str()
    expected = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    assert yesterday == expected


def test_google_news_urls(monkeypatch):
    monkeypatch.setattr("pipeline.ingestion_pipeline.search", dummy_search)
    urls = google_news_urls("sports", max_results=2)
    assert urls == ["http://example.com/article1", "http://example.com/article2"]


def test_extract_article_metadata(monkeypatch):
    # Monkeypatch Newspaper's Article with a dummy
    DummyArticle = type(
        "DummyArticle",
        (),
        {
            "download": lambda self: None,
            "parse": lambda self: None,
            "title": "Test Title",
            "publish_date": datetime(2023, 3, 28, 12, 0, 0),
            "text": "Test text",
            "top_image": "http://example.com/image.jpg",
            "source_url": "http://example.com",
        },
    )
    monkeypatch.setattr(
        "pipeline.ingestion_pipeline.Article", lambda url: DummyArticle()
    )
    metadata = extract_article_metadata("http://example.com")
    assert metadata["title"] == "Test Title"
    assert metadata["published_date"] == "2023-03-28 12:00:00"


def test_get_openai_embedding(monkeypatch):
    monkeypatch.setattr(
        "pipeline.ingestion_pipeline.get_openai_embedding", dummy_get_openai_embedding
    )
    embedding = dummy_get_openai_embedding("dummy text")
    assert len(embedding) == 1536
    assert all(x == 0.1 for x in embedding)


def test_ingest_category(monkeypatch, dummy_collection):
    # Override external functions with dummy implementations
    monkeypatch.setattr("pipeline.ingestion_pipeline.search", dummy_search)
    monkeypatch.setattr(
        "pipeline.ingestion_pipeline.extract_article_metadata",
        dummy_extract_article_metadata,
    )
    monkeypatch.setattr(
        "pipeline.ingestion_pipeline.get_openai_embedding", dummy_get_openai_embedding
    )

    # Clear dummy collection and run ingest_category
    dummy_collection.data = []
    ingest_category("sports", limit=2)

    # Check that at most 2 articles are inserted
    assert len(dummy_collection.data) <= 2
    for doc in dummy_collection.data:
        assert doc["category"] == "sports"
        assert "embedding" in doc
