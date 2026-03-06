## Local RAG PDF Chatbot ##

A fully local Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF and ask questions about its content.
The system extracts text from the document, retrieves relevant sections using semantic search, and generates answers using a local language model — without any external APIs or cloud services.

This project demonstrates how to build a secure, privacy-first document question answering system using open-source tools.

# 🚀 Features

Upload any PDF document

Ask questions in natural language

Semantic search over document content

Local LLM answer generation

View source document chunks used for answers

Adjustable retrieval settings

Fully local (no API keys required)

## Architecture ##

The chatbot follows a Retrieval-Augmented Generation (RAG) pipeline.

PDF Upload
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Embeddings Generation
   ↓
Vector Storage (FAISS)
   ↓
User Query
   ↓
Semantic Retrieval (Top-K chunks)
   ↓
Local LLM (Flan-T5)
   ↓
Generated Answer
