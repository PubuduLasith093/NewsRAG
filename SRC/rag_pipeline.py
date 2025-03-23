import os
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")


def rag_answer_question(
    query: str,
    namespace: str = "news_data.articles",
    index_name: str = "news_index",
    top_k: int = 5,
) -> Tuple[str, List[Dict]]:
    """
    Perform RAG to answer a question using MongoDB vector store and return matching sources.

    Args:
        query (str): User question
        namespace (str): MongoDB namespace in format 'db.collection'
        index_name (str): MongoDB Atlas vector index name
        top_k (int): Number of top documents to retrieve

    Returns:
        Tuple[str, List[Dict]]: Answer string and a list of matching sources with metadata
    """
    # Initialize embeddings and language model
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(api_key=OPENAI_API_KEY)

    # Setup vector store from MongoDB
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=MONGO_URI,
        namespace=namespace,
        embedding=embeddings,
        index_name=index_name,
    )

    # Configure retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )

    # Setup RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    # Invoke the chain
    response = qa_chain.invoke({"query": query})

    # Format sources
    sources = []
    for doc in response["source_documents"]:
        metadata = doc.metadata
        sources.append(
            {
                "title": metadata.get("title", "No Title"),
                "source": metadata.get("source", "Unknown"),
                "url": metadata.get("url", "N/A"),
                "published_date": metadata.get("published_date", "N/A"),
            }
        )

    return response["result"], sources


# if __name__ == "__main__":
#     question = "Will BMW’s Neue Klasse range of electric vehicles be available in Australia before 2026?"

#     answer, sources = rag_answer_question(query=question)

#     print("📌 Answer:")
#     print(answer)

#     print("\n📰 Sources:")
#     for s in sources:
#         print(f"- {s['title']} ({s['source']}) — {s['url']}")
