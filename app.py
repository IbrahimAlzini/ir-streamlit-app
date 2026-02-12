
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed document embeddings (Assuming embeddings.npy and documents.txt exist)
embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]

def retrieve_top_k(query_embedding, embeddings, k=10):
    """Retrieve top-k most similar documents using cosine similarity."""
    sims = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = sims.argsort()[-k:][::-1]
    return [(documents[i], float(sims[i])) for i in top_k_indices]

def get_query_embedding(query, dim):
    """
    Placeholder embedding generator.
    Replace this with a real embedding model for 'fine tuning' later.
    """
    rng = np.random.default_rng(abs(hash(query)) % (2**32))
    return rng.random(dim).astype(np.float32)

# Streamlit UI
st.title("Information Retrieval using Document Embeddings")

query = st.text_input("Enter your query:")
k = st.slider("Top K results", min_value=1, max_value=min(20, len(documents)), value=min(10, len(documents)))

if st.button("Search"):
    if not query.strip():
        st.warning("Please type a query first.")
    else:
        query_embedding = get_query_embedding(query, embeddings.shape[1])
        results = retrieve_top_k(query_embedding, embeddings, k=k)

        st.write(f"### Top {k} Relevant Documents:")
        for doc, score in results:
            st.write(f"- **{doc}** (Score: {score:.4f})")
