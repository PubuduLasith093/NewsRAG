import os
import umap
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from SRC.utils.fetch_articles import fetch_articles_with_embeddings

# Load MongoDB URI from .env
load_dotenv()
# MONGO_URI = os.getenv("MONGO_URI")


def perform_duplicate_clustering(articles: list, threshold: float = 0.85):
    """
    Perform duplicate detection using Agglomerative Clustering based on cosine similarity.

    Args:
        articles (list): List of article documents with an 'embedding' field.
        threshold (float): Cosine similarity threshold to consider two articles duplicates.
                           A higher value means articles must be very similar.

    Returns:
        np.array: Cluster labels for each article.
    """
    # Extract embeddings from articles
    embeddings = np.array([doc["embedding"] for doc in articles])

    # Dimensionality reduction using UMAP (with cosine metric)
    # reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric='cosine')
    # reduced_embeddings = reducer.fit_transform(embeddings)

    # Compute cosine similarity matrix and then a distance matrix
    sim_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - sim_matrix  # cosine distance

    # Agglomerative clustering with a precomputed distance matrix
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1
        - threshold,  # Adjust so that only very similar articles are clustered together
    )

    labels = clustering.fit_predict(distance_matrix)
    return labels


def update_articles_with_clusters(
    db, collection_name: str, articles: list, labels: np.array
):
    """
    Update each article in MongoDB with its duplicate cluster ID.

    Args:
        db: MongoDB database instance.
        collection_name (str): Name of the collection.
        articles (list): List of article documents.
        labels (np.array): Cluster labels corresponding to each article.
    """
    collection = db[collection_name]
    for doc, label in zip(articles, labels):
        collection.update_one(
            {"_id": doc["_id"]}, {"$set": {"duplicate_cluster_id": int(label)}}
        )
    print(f"Updated {len(articles)} articles with duplicate_cluster_id.")


def run_duplicate_detection(mongo_client):
    # Connect to MongoDB
    db = mongo_client["news_data"]
    collection_name = "articles"

    # Step 1: Fetch articles that contain embeddings
    articles = fetch_articles_with_embeddings(db, collection_name)
    if not articles:
        print("No articles found with embeddings.")
        return

    print(f"Fetched {len(articles)} articles from MongoDB.")

    # Step 2: Perform duplicate detection clustering
    labels = perform_duplicate_clustering(articles, threshold=0.85)

    # Step 3: Update articles with the computed duplicate cluster IDs
    update_articles_with_clusters(db, collection_name, articles, labels)

    # Optional: Print cluster groups for review
    clusters = {}
    for doc, label in zip(articles, labels):
        clusters.setdefault(label, []).append(doc["_id"])

    print("Duplicate clusters found:")
    for cluster_id, doc_ids in clusters.items():
        print(f"Cluster {cluster_id}: {doc_ids}")


# if __name__ == "__main__":
#     run_duplicate_detection()
