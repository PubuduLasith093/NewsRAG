import os
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["news_data"]
collection = db["articles"]

st.set_page_config(page_title="News Highlights", layout="wide")

st.title("📰 Daily Highlights")
st.sidebar.header("🧭 Filter")

# Sidebar filters
categories = [
    "sports",
    "finance",
    "music",
    "lifestyle",
    "technology",
    "politics",
    "health",
]
selected_category = st.sidebar.selectbox("Select Category", categories)

today = datetime.now()
yesterday = today - timedelta(days=1)
selected_date = st.sidebar.date_input("Published After", yesterday.date())


def fetch_featured_articles(category: str):
    """
    Fetch featured articles for the given category and date.
    Assumes featured articles have the field 'is_featured': True.
    """
    query = {
        "predicted_topic": category,
        "is_featured": True,
        "ingested_at": {"$gte": selected_date.strftime("%Y-%m-%d")},
    }
    pipeline = [
        {"$match": query},
        {"$sort": {"featured_at": -1}},  # Most recent featured first
        {"$limit": 3},
    ]
    return list(collection.aggregate(pipeline))


def fetch_non_featured_articles(category: str):
    """
    Fetch non-featured (or not marked as featured) articles for the given category and date.
    Applies additional scoring based on keywords and duplicate clustering.
    """
    query = {
        "predicted_topic": category,
        "$or": [{"is_featured": {"$exists": False}}, {"is_featured": False}],
        "ingested_at": {"$gte": selected_date.strftime("%Y-%m-%d")},
    }
    pipeline = [
        {"$match": query},
        {
            "$addFields": {
                "score": {
                    "$add": [
                        {
                            "$cond": [
                                {
                                    "$regexMatch": {
                                        "input": "$title",
                                        "regex": "breaking|exclusive|confirmed",
                                        "options": "i",
                                    }
                                },
                                5,
                                0,
                            ]
                        },
                        {
                            "$cond": [
                                {"$ifNull": ["$duplicate_cluster_id", False]},
                                1,
                                0,
                            ]
                        },
                    ]
                }
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": 12},
    ]
    return list(collection.aggregate(pipeline))


# Fetch data
featured_articles = fetch_featured_articles(selected_category)
other_articles = fetch_non_featured_articles(selected_category)

if not featured_articles and not other_articles:
    st.warning("No articles found for the selected filters.")
else:
    # Display Featured Section
    if featured_articles:
        st.subheader("🌟 Featured")
        for article in featured_articles:
            st.markdown(f"**📰 {article.get('title', 'Untitled')}**")
            st.markdown(
                f"*Source:* [{article.get('source', 'Unknown')}]({article.get('url', '#')}) | *Published:* {article.get('published_date', 'N/A')}"
            )
            st.markdown("---")
    else:
        st.info("No featured articles found.")

    # Display Other Highlights as Cards (3 per row)
    if other_articles:
        st.subheader("📚 All Highlights")
        cols = st.columns(3)
        for i, article in enumerate(other_articles):
            with cols[i % 3]:
                st.markdown("📰 **" + article.get("title", "Untitled") + "**")
                st.caption(
                    f"{article.get('source', 'Unknown')} | {article.get('published_date', 'N/A')}"
                )
                # Show a short preview of the text (first 150 characters)
                st.write(article.get("text", "")[:150] + "...")
    else:
        st.info("No other articles found.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + MongoDB")
