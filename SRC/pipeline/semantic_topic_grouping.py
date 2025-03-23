import os
import numpy as np
import umap
import hdbscan
from dotenv import load_dotenv
from SRC.utils.fetch_articles import fetch_articles_with_embeddings


def reduce_embeddings(
    embeddings: np.ndarray,
    n_components: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """
    Reduce the high-dimensional embeddings to a lower dimension using UMAP.

    Args:
        embeddings (np.ndarray): High-dimensional embeddings.
        n_components (int): Target number of dimensions (default 50).
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance for UMAP.

    Returns:
        np.ndarray: Reduced embeddings.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="cosine",
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced


def perform_topic_grouping(
    reduced_embeddings: np.ndarray, min_cluster_size: int = 5, min_samples: int = 1
) -> np.ndarray:
    """
    Perform semantic topic grouping using HDBSCAN on the reduced embeddings.

    Args:
        reduced_embeddings (np.ndarray): Embeddings after dimensionality reduction.
        min_cluster_size (int): Minimum size of clusters.
        min_samples (int): Minimum samples in a neighborhood for a point to be considered core.

    Returns:
        np.ndarray: Cluster labels for each article.
    """
    # HDBSCAN uses 'euclidean' distance by default which works well on UMAP output.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean"
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    return labels


def update_articles_with_topic_clusters(
    db, collection_name: str, articles: list, labels: np.ndarray
):
    """
    Update each article in MongoDB with its semantic topic cluster ID.

    Args:
        db: MongoDB database instance.
        collection_name (str): The collection name.
        articles (list): List of article documents.
        labels (np.ndarray): Cluster labels corresponding to each article.
    """
    collection = db[collection_name]
    for doc, label in zip(articles, labels):
        # Use None for noise (HDBSCAN labels noise as -1)
        topic_cluster_id = int(label) if label != -1 else None
        collection.update_one(
            {"_id": doc["_id"]}, {"$set": {"topic_cluster_id": topic_cluster_id}}
        )
    print(f"Updated {len(articles)} articles with topic_cluster_id.")


def run_semantic_grouping(mongo_client):
    # Connect to MongoDB
    db = mongo_client["news_data"]
    collection_name = "articles"

    # Fetch articles with embeddings
    articles = fetch_articles_with_embeddings(db, collection_name)
    if not articles:
        print("No articles found with embeddings.")
        return
    print(f"Fetched {len(articles)} articles from MongoDB.")

    # Extract embeddings as a NumPy array
    embeddings = np.array([doc["embedding"] for doc in articles])

    # # Reduce dimensionality using UMAP for better clustering
    # reduced_embeddings = reduce_embeddings(
    #     embeddings, n_components=50, n_neighbors=15, min_dist=0.1
    # )

    # Perform topic grouping with HDBSCAN
    labels = perform_topic_grouping(
        embeddings,
        min_cluster_size=2,
        min_samples=1,
    )

    # Update MongoDB with the topic cluster labels
    update_articles_with_topic_clusters(db, collection_name, articles, labels)

    # Print summary of topic clusters
    clusters = {}
    for doc, label in zip(articles, labels):
        clusters.setdefault(label, []).append(doc["_id"])
    print("Topic clusters found:")
    for cluster_id, doc_ids in clusters.items():
        print(f"Cluster {cluster_id}: {doc_ids}")


# if __name__ == "__main__":
#     run_semantic_grouping()
