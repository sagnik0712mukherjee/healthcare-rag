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
| Frontend Deployment | Streamlit Cloud |
| Backend Deployment | Railway |
| Database Deployment | Railway Postgres plugin |

---

## Repository Structure

```
healthcare-rag/
|
|-- streamlit.py                      # Streamlit frontend (must stay in root)
|-- start.sh                          # Railway startup script (builds FAISS index on first boot)
|-- requirements.txt                  # All Python dependencies
|-- railway.json                      # Railway deployment config
|-- Procfile                          # Process definition (fallback)
|-- README.md                         # This file
|
|-- config/                           # Application configuration (outside src)
|   |-- __init__.py
|   |-- settings.py                   # Reads all settings from .env
|
|-- data/
|   |-- raw/                          # Original MultiCaRe dataset files (Git LFS)
|       |-- cases.parquet             # 93,816 clinical cases
|       |-- captions_and_labels.csv   # 130,791 image captions + labels
|       |-- metadata.parquet          # Article metadata
|
|-- src/                              # All application source code
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
|   |   |-- short_term_memory.py      # In-memory session history (+ pinned chunks)
|   |   |-- long_term_memory.py       # DB-backed persistent memory
|   |
|   |-- monitoring/                   # Token usage tracking
|   |   |-- __init__.py
|   |   |-- token_tracker.py          # Budget checks per user
|   |   |-- usage_logger.py           # Logs usage to PostgreSQL
|   |
|   |-- rag/                          # Core RAG logic
|   |   |-- __init__.py
|   |   |-- retriever.py              # FAISS similarity search (+ pinned chunk injection)
|   |   |-- generator.py              # OpenAI prompt + response generation
|   |   |-- pipeline.py               # Orchestrates the full RAG flow
|   |
|   |-- utils/                        # Shared helper functions
|       |-- __init__.py
|       |-- helpers.py                # Text cleaning, risk scoring, timestamps
|
|-- vectorstore/
|   |-- faiss_index/                  # Auto-generated by build_faiss_index.py
|       |-- index.bin                 # FAISS binary index (114MB, Git LFS)
|       |-- metadata.json             # Chunk metadata (Git LFS)
|       |-- summary.json              # Index build summary
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sagnik0712mukherjee/healthcare-rag.git
cd healthcare-rag
```

### 2. Install Git LFS (required for data files)

```bash
git lfs install
git lfs pull
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

```bash
cp .env.example .env
# Open .env and fill in:
#   OPENAI_API_KEY=sk-...
#   DATABASE_URL=sqlite:///./healthcare_rag.db   # local dev
#   SECRET_KEY=your-secret-key
```

### 6. Build the FAISS Index (first time only)

```bash
PYTHONPATH=. python3 -m src.ingestion.build_faiss_index --max-cases 5000 --max-images 5000
```

### 7. Run the Backend

```bash
uvicorn src.api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 8. Run the Frontend

```bash
streamlit run streamlit.py
# Opens at: http://localhost:8501
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/auth/register` | Create a new user account |
| POST | `/api/v1/auth/login` | Log in and receive JWT token |
| POST | `/api/v1/query` | Submit a medical question |
| GET | `/api/v1/history` | Get conversation history |
| GET | `/api/v1/me` | Get current user info |
| GET | `/api/v1/admin/users` | List all users (admin only) |
| GET | `/api/v1/admin/review` | Get flagged responses (admin only) |
| POST | `/api/v1/admin/review/{id}` | Approve/reject flagged response |
| GET | `/api/v1/admin/cache/stats` | Cache statistics (admin only) |
| DELETE | `/api/v1/admin/cache` | Clear query cache (admin only) |

---

## Deployment

### Backend → Railway

1. Push code to GitHub
2. Create Railway project → connect GitHub repo
3. Add PostgreSQL plugin
4. Set environment variables (see `.env.example`)
5. Railway uses `railway.json` — `start.sh` runs on first boot, downloads data from Zenodo, builds FAISS index, then starts the API

### Frontend → Streamlit Cloud

1. Go to https://share.streamlit.io
2. Connect GitHub repo, set main file: `streamlit.py`
3. Add secret: `API_BASE_URL = "https://your-railway-url/api/v1"`

---

## Guardrails

**Input Guardrails** check for self-harm queries, illegal drug requests, and prompt injections before any LLM call.

**Output Guardrails** remove diagnostic/prescriptive language and append a standard disclaimer to every response:

> *"This information is for educational purposes only. Please consult a qualified healthcare professional for medical advice, diagnosis, or treatment."*

---

## Token Usage Limits

Each user has a default limit of 100,000 tokens (~$0.02 with GPT-4o-mini). Usage is tracked per request in PostgreSQL. Admins can view and adjust limits via the admin panel.

---

## Evaluation

```bash
python src/evaluation/ragas_eval.py
```

Produces RAGAS scores for Faithfulness, Answer Relevance, and Context Recall.

---

## Dataset Citation

> Nievas Offidani, M.; Roffet, F.; González Galtier, M.C.; Massiris, M.; Delrieux, C.
> An Open-Source Clinical Case Dataset for Medical Image Classification and Multimodal AI Applications.
> *Data* 2025, 10, 123. https://doi.org/10.3390/data10080123

Dataset DOI: https://doi.org/10.5281/zenodo.10079369 · License: CC-BY-NC-SA

---

## Author

Created by **Sagnik Mukherjee** · [GitHub](https://github.com/sagnik0712mukherjee)