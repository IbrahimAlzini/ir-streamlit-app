# Information Retrieval using Document Embeddings (Streamlit App)

## Overview
This project demonstrates a simple Information Retrieval (IR) system using document embeddings and cosine similarity.

Users can enter a query, and the system retrieves the top-K most relevant documents.

The app is built using:
- Streamlit
- NumPy
- Scikit-learn

## How It Works
1. Documents are stored in `documents.txt`
2. Precomputed embeddings are stored in `embeddings.npy`
3. Query embeddings are generated (placeholder)
4. Cosine similarity is used to rank documents
5. Top-K results are displayed in the Streamlit UI

## Files
- `app.py` – Streamlit application
- `documents.txt` – Dataset
- `embeddings.npy` – Precomputed document embeddings
- `requirements.txt` – Required dependencies

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
