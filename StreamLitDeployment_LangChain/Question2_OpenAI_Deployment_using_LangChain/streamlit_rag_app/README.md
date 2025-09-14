# Streamlit RAG App

This repository contains a simple Streamlit RAG (Retrieval-Augmented Generation) application using LangChain + OpenAI + FAISS.

## Quickstart (local)
1. Set environment variable: `export OPENAI_API_KEY=your_key`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run streamlit_app.py`

## Quickstart (Docker)
1. Build: `docker build -t streamlit-rag .`
2. Run: docker run -p 8501:8501 streamlit-rag
