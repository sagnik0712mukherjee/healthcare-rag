# Healthcare RAG System

A production-style Retrieval Augmented Generation (RAG) application for healthcare queries, built on the **MultiCaRe dataset** — an open-source collection of 93,816 clinical cases and 130,791 medical images derived from PubMed Central case reports.

---

## What This System Does

- Allows patients and healthcare users to ask medical questions in natural language
- Retrieves relevant clinical cases and image captions from a vector database (FAISS)
- Generates safe, contextual responses using OpenAI GPT-4o-mini
- Maintains conversation context across follow-up questions
- Enforces healthcare-specific safety guardrails on every response
- Tracks token usage per user and enforces spending limits
- Supports optional human review of flagged responses

---

## Architecture Overview

```
User Query (Streamlit UI)
        |
        v
FastAPI Backend (/src/api)
        |
        |-- Query Cache Check (PostgreSQL)
        |       If cached: return immediately
        |
        |-- Input Guardrails
        |       Block unsafe queries
        |
        |-- Short-Term Memory
        |       Inject conversation history
        |
        |-- FAISS Retriever
        |       Embed query -> similarity search
        |       Returns: clinical case chunks + image captions
        |
        |-- LLM Generator (OpenAI)
        |       Constructs prompt with context
        |       Calls GPT-4o-mini
        |
        |-- Output Guardrails
        |       Strip diagnosis/prescription language
        |       Append medical disclaimer
        |
        |-- Token Monitoring
        |       Log usage to PostgreSQL
        |       Check user has not exceeded limit
        |
        v
Response returned to user
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend API | FastAPI |
| Vector Database | FAISS (Facebook AI Similarity Search) |
| LLM Provider | OpenAI (GPT-4o-mini) |
| Embeddings | OpenAI (text-embedding-3-small) |
| Relational Database | PostgreSQL |
| Storage | Local disk or AWS S3 (free tier) |
| Frontend Deployment | Streamlit Cloud |
| Backend Deployment | Railway |
| Database Deployment | Railway Postgres plugin |

---

## Repository Structure

```
healthcare-rag/
|
|-- streamlit.py                      # Streamlit frontend (must stay in root)
|-- requirements.txt                  # All Python dependencies
|-- .env.example                      # Template for environment variables
|-- railway.json                      # Railway deployment config
|-- Procfile                          # Process definition for Railway
|-- README.md                         # This file
|
|-- config/                           # Application configuration (outside src)
|   |-- __init__.py
|   |-- settings.py                   # Reads all settings from .env
|
|-- data/                             # Dataset files (not committed to git)
|   |-- images/                       # Medical images (.webp format)
|   |-- processed/                    # Chunked + metadata JSON files
|   |-- raw/                          # Original MultiCaRe .parquet/.csv files
|
|-- src/                              # All application source code lives here
|   |-- __init__.py
|   |
|   |-- api/                          # FastAPI backend
|   |   |-- __init__.py
|   |   |-- main.py                   # App factory, startup, CORS
|   |   |-- routes.py                 # All API route handlers
|   |   |-- schemas.py                # Pydantic request/response models
|   |
|   |-- caching/                      # Query result caching
|   |   |-- __init__.py
|   |   |-- query_cache.py            # Hash-based response cache in PostgreSQL
|   |
|   |-- database/                     # Database layer
|   |   |-- __init__.py
|   |   |-- db.py                     # Engine, session factory, get_db()
|   |   |-- models.py                 # SQLAlchemy ORM table definitions
|   |
|   |-- evaluation/                   # RAG quality evaluation
|   |   |-- __init__.py
|   |   |-- ragas_eval.py             # RAGAS metrics runner
|   |
|   |-- guardrails/                   # Safety filters
|   |   |-- __init__.py
|   |   |-- input_guardrails.py       # Blocks unsafe input queries
|   |   |-- output_guardrails.py      # Cleans output, appends disclaimer
|   |
|   |-- ingestion/                    # Data loading and indexing pipeline
|   |   |-- __init__.py
|   |   |-- load_cases.py             # Loads cases.parquet
|   |   |-- load_images.py            # Loads captions_and_labels.csv
|   |   |-- chunking.py               # Splits case text into chunks
|   |   |-- embeddings.py             # Generates OpenAI embeddings
|   |   |-- build_faiss_index.py      # Runs the full ingestion pipeline
|   |
|   |-- memory/                       # Conversation memory
|   |   |-- __init__.py
|   |   |-- short_term_memory.py      # In-memory session history
|   |   |-- long_term_memory.py       # DB-backed persistent memory
|   |
|   |-- monitoring/                   # Token usage tracking
|   |   |-- __init__.py
|   |   |-- token_tracker.py          # Budget checks per user
|   |   |-- usage_logger.py           # Logs usage to PostgreSQL
|   |
|   |-- rag/                          # Core RAG logic
|   |   |-- __init__.py
|   |   |-- retriever.py              # FAISS similarity search
|   |   |-- generator.py              # OpenAI prompt + response generation
|   |   |-- pipeline.py               # Orchestrates the full RAG flow
|   |
|   |-- utils/                        # Shared helper functions
|   |   |-- __init__.py
|   |   |-- helpers.py                # Text cleaning, risk scoring, timestamps
|   |
|   |-- vectorstore/                  # FAISS index files (auto-generated)
|       |-- __init__.py
|       |-- faiss_index/              # Created by build_faiss_index.py
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/healthcare-rag.git
cd healthcare-rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Open .env and fill in your actual values:
#   - OPENAI_API_KEY
#   - DATABASE_URL
```

### 5. Set Up the Database

```bash
# Make sure PostgreSQL is running locally, then:
python -c "from src.database.db import create_tables; create_tables()"
```

### 6. Download the MultiCaRe Dataset

```bash
# Option A: Use the multiversity Python package (recommended)
pip install multiversity
python -c "
from multiversity.multicare_creator import MulticareCreator
mc = MulticareCreator(email='your@email.com', api_key='your-ncbi-key')
mc.download_dataset()
"

# Option B: Download directly from Zenodo
# https://zenodo.org/records/14994046
# Place files in data/raw/
```

### 7. Build the FAISS Index

This step reads the dataset, creates text chunks, generates embeddings, and saves the FAISS index. It will take several minutes depending on how much data you ingest.

```bash
python src/ingestion/build_faiss_index.py
```

### 8. Run the FastAPI Backend

```bash
uvicorn src.api.main:app --reload --port 8000
# API docs available at: http://localhost:8000/docs
```

### 9. Run the Streamlit Frontend

```bash
# In a new terminal window:
streamlit run streamlit.py
# Opens at: http://localhost:8501
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Submit a medical question |
| GET | `/health` | Health check |
| GET | `/admin/users` | List all users and their token usage |
| GET | `/admin/review` | Get responses pending human review |
| POST | `/admin/review/{id}/approve` | Approve a flagged response |
| POST | `/admin/review/{id}/reject` | Reject a flagged response |
| POST | `/auth/register` | Create a new user account |
| POST | `/auth/login` | Log in and receive a token |

---

## Deployment

### Deploy Backend to Railway

1. Push your code to GitHub
2. Create a new Railway project at https://railway.app
3. Connect your GitHub repository
4. Add a PostgreSQL plugin from the Railway dashboard
5. Set all environment variables from `.env.example` in Railway's Variables tab
6. Railway will automatically build and deploy using `railway.json`

### Deploy Frontend to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Connect your GitHub repository
3. Set the main file path to: `streamlit.py`
4. Add your environment variables in Streamlit Cloud's Secrets section
5. Set `FASTAPI_URL` to your Railway backend URL

---

## Environment Variables Reference

See `.env.example` for the complete list of required and optional variables with explanations.

---

## Guardrails

This system enforces two layers of safety:

**Input Guardrails** — Before sending a query to the LLM, the system checks for:
- Self-harm related queries
- Requests for illegal drug information
- Attempts to get the AI to prescribe medication directly

**Output Guardrails** — Before returning a response, the system:
- Removes language that sounds like a medical diagnosis
- Removes language that sounds like a prescription
- Appends a standard medical disclaimer to every response

Every response ends with:
> "This information is for educational purposes only. Please consult a qualified healthcare professional for medical advice, diagnosis, or treatment."

---

## Token Usage Limits

Each user has a token usage limit (default: 100,000 tokens, approximately $0.02 with GPT-4o-mini).

- Token usage is tracked per request and stored in PostgreSQL
- Users who exceed their limit receive a clear error message
- Admins can view and adjust limits via the admin panel

---

## Evaluation

Run RAG quality evaluation using RAGAS:

```bash
python src/evaluation/ragas_eval.py
```

This produces scores for:
- **Faithfulness**: Does the answer stay true to the retrieved context?
- **Answer Relevance**: Does the answer actually address the question?
- **Context Recall**: Did retrieval find the right information?

---

## Dataset Citation

If you use this system, please cite the MultiCaRe dataset:

> Nievas Offidani, M.; Roffet, F.; González Galtier, M.C.; Massiris, M.; Delrieux, C.
> An Open-Source Clinical Case Dataset for Medical Image Classification and Multimodal AI Applications.
> Data 2025, 10, 123. https://doi.org/10.3390/data10080123

Dataset DOI: https://doi.org/10.5281/zenodo.10079369
Dataset License: CC-BY-NC-SA

---

## Author

Created with passion by **Sagnik Mukherjee**.
- [GitHub Profile](https://github.com/sagnik0712mukherjee)