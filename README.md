# IntelliAssist AI – Smart Document AI Assistant

IntelliAssist AI is an AI-powered document assistant that allows users to upload documents and interact with them through a conversational interface.

## Features

- Upload PDF, DOCX and TXT documents
- Ask questions about uploaded documents
- Retrieval-Augmented Generation (RAG)
- Semantic search using Hugging Face embeddings
- FAISS vector database
- Hybrid search using FAISS and TF-IDF
- Context-aware conversational chat
- Document summarization
- Study notes generation
- Quiz generation
- Sentiment and intent analysis
- Source/page citations
- Chat history and export
- Streamlit web interface

## Technology Stack

- Python
- Streamlit
- LangChain
- Google Gemini
- Hugging Face Sentence Transformers
- FAISS
- TF-IDF
- Scikit-learn
- VADER Sentiment Analysis

## Architecture

```text
Document Upload
       ↓
Text Extraction
       ↓
Text Chunking
       ↓
Hugging Face Embeddings + TF-IDF
       ↓
FAISS Vector Database
       ↓
Hybrid Retrieval
       ↓
Google Gemini
       ↓
Answer + Source Citations