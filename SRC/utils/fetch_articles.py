def fetch_articles_with_embeddings(db, collection_name: str):
    """
    Fetch all articles from MongoDB that have an 'embedding' field.

    Args:
        db: MongoDB database instance.
        collection_name (str): Name of the collection to query.

    Returns:
        List of documents (articles).
    """
    collection = db[collection_name]
    return list(collection.find({"embedding": {"$exists": True}}))
