import numpy as np
import pytest
from pipeline.duplicate_detection import (
    perform_duplicate_clustering,
    update_articles_with_clusters,
)

# Create dummy articles with embeddings and _id fields
DUMMY_ARTICLES = [
    {"_id": "a1", "embedding": [0.1, 0.1, 0.1]},
    {"_id": "a2", "embedding": [0.1, 0.1, 0.1]},  # duplicate of a1
    {"_id": "a3", "embedding": [0.9, 0.0, 0.0]},  # different
    {"_id": "a4", "embedding": [0.9, 0.0, 0.0]},  # duplicate of a3
    {
        "_id": "a5",
        "embedding": [0.2, 0.2, 0.2],
    },  # somewhat similar to a1/a2 but not identical
]


# For testing update, create a dummy collection class to simulate MongoDB
class DummyCollection:
    def __init__(self):
        self.data = {}

    def update_one(self, query, update):
        _id = query.get("_id")
        self.data[_id] = update["$set"]

    def find(self, query):
        # Return items with duplicate_cluster_id if present
        return [{"_id": _id, **vals} for _id, vals in self.data.items()]


# Dummy db object with a 'articles' collection attribute
class DummyDB:
    def __init__(self):
        self.articles = DummyCollection()


@pytest.fixture
def dummy_db():
    return DummyDB()


def test_perform_duplicate_clustering():
    # Using threshold 0.85: articles a1 and a2 should cluster together,
    # articles a3 and a4 together, and a5 might join with a1/a2 if similarity is high enough.
    labels = perform_duplicate_clustering(DUMMY_ARTICLES, threshold=0.85)
    # Check that a1 and a2 are in the same cluster.
    assert labels[0] == labels[1]
    # Check that a3 and a4 are in the same cluster.
    assert labels[2] == labels[3]
    # a5 could be in a cluster with a1/a2 or separate depending on cosine distance.
    # For cosine similarity, cosine([0.1,0.1,0.1], [0.2,0.2,0.2]) = 1 (if normalized) or similar.
    # We won't enforce a strict check on a5; just ensure we get labels of same length.
    assert len(labels) == len(DUMMY_ARTICLES)


def test_update_articles_with_clusters(dummy_db):
    # Use dummy labels for the dummy articles
    dummy_labels = np.array([0, 0, 1, 1, 0])
    # Call update function with dummy db
    update_articles_with_clusters(dummy_db, "articles", DUMMY_ARTICLES, dummy_labels)

    # Verify that each dummy article now has the correct duplicate_cluster_id set in dummy_db
    updated_docs = dummy_db.articles.find({})
    results = {doc["_id"]: doc["duplicate_cluster_id"] for doc in updated_docs}

    # Expected: a1 and a2 in cluster 0, a3 and a4 in cluster 1, a5 in cluster 0
    expected = {"a1": 0, "a2": 0, "a3": 1, "a4": 1, "a5": 0}
    assert results == expected
