import os
import streamlit as st
import subprocess
from pymongo import MongoClient
from SRC.pipeline.ingestion_pipeline import run_ingestion_pipeline
from SRC.pipeline.duplicate_detection import run_duplicate_detection
from SRC.pipeline.semantic_topic_grouping import run_semantic_grouping
from SRC.pipeline.zero_shot_topic_clustering import zeroshot_topic_clustering
from SRC.pipeline.highlight_news import run_highlight

st.set_page_config(page_title="News MAS Pipeline", page_icon="🗞️", layout="centered")

st.title("🧠 Welcome to NewsMAS – Your Daily AI-Powered News Assistant")
st.sidebar.success("Choose a page to get started!")
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
# st.title("🧠 News MAS: Run Full Pipeline")

if st.button("🚀 Run End-to-End Pipeline"):
    with st.spinner("Fetching and processing news... this may take a few minutes"):
        try:
            # # Step 1: Ingest news
            # # run_ingestion_pipeline(mongo_client)

            # # Step 2: Detect duplicates
            # run_duplicate_detection(mongo_client)

            # # Step 3: Group topics
            # run_semantic_grouping(mongo_client)
            # zeroshot_topic_clustering(mongo_client)

            # Step 4: Highlight News Selection
            run_highlight(mongo_client)

            st.success("✅ News ingestion and processing completed!")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Error occurred: {e}")
        except Exception as ex:
            st.error(f"🔥 Unexpected error: {ex}")

st.markdown("---")
st.markdown(
    """
This dashboard lets you:
- Run the full pipeline with a single click
- Automatically ingest news, detect duplicates, cluster topics, and assign labels
- Data is saved to MongoDB with embeddings and metadata
"""
)
