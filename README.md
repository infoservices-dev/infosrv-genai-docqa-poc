# Infoservices Agentic RAG System - POC

Enterprise-grade Document Intelligence Platform powered by AWS Bedrock and ChromaDB.

## Overview

This POC demonstrates an advanced Retrieval-Augmented Generation (RAG) system that combines document processing, semantic search, and AI-powered question answering. Built with enterprise scalability and extensibility in mind.

## Features

- **Multi-Format Document Support**: PDF, TXT, Markdown
- **AI-Powered Q&A**: Context-aware responses using AWS Bedrock Claude
- **Semantic Search**: Vector-based document retrieval with relevance scoring
- **Enterprise UI**: Professional Streamlit interface with real-time metrics
- **Extensible Architecture**: Plugin-based design for embeddings, storage, and loaders
- **Cloud-Native**: AWS Bedrock integration with ChromaDB vector store

## Architecture

```
infosrv-genai-docqa-poc/
├── src/                    # Source code
│   ├── agents/            # Agentic components
│   ├── ingestion/         # Document processing & ingestion
│   ├── embeddings/        # Embedding generation
│   ├── retrieval/         # Vector search & retrieval
│   ├── qa/                # Question answering logic
│   └── utils/             # Shared utilities
├── data/                  # Document storage
│   ├── raw/              # Source documents
│   └── processed/        # Processed documents
├── notebooks/             # Jupyter notebooks for experimentation
├── tests/                 # Test suite
├── config/                # Configuration files
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
└── README.md
```

## Getting Started

1. Clone repository:

```bash
git clone https://github.com/infoservices-dev/infosrv-genai-docqa-poc
cd infosrv-genai-docqa-poc
```

2. Backend setup:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Configure AWS credentials:

```bash
aws configure
```

4. Run the application:

```bash
python src/main.py
```

5. Run tests:

```bash
pytest tests/
```

