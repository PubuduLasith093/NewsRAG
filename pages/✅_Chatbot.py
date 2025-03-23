import streamlit as st
from SRC.rag_pipeline import rag_answer_question

st.set_page_config(page_title="Ask the News", layout="centered")
st.title("💬 Ask the NewsBot")
st.write("Ask questions based on latest news articles:")

query = st.text_input(
    "Type your question", placeholder="e.g., What's happening in the finance sector?"
)
if query:
    with st.spinner("Retreiving..."):
        answer, sources = rag_answer_question(query)
        st.success(answer)
        st.markdown("### 📰 Sources")
        for src in sources:
            st.markdown(f"- [{src['title']}]({src['url']}) ({src['source']})")
