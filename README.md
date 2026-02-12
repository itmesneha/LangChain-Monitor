# LangChain-Monitor

A Retrieval-Augmented Generation (RAG) system that monitors LangChain GitHub issues, extracts business and technical insights, stores them as vector embeddings in Milvus, and serves intelligent answers via a FastAPI endpoint. Optionally fine-tunes a small LLM (Phi-3) on the generated insights for grounded response generation.

## Architecture

```
GitHub Issues ──> Ingestion ──> Preprocessing ──> Insight Extraction
                                                        │
                                                        ▼
                              API  <──  Milvus Vector DB  <──  Embeddings (BGE-m3)
                                                        
```

## Project Structure

```
├── src/
│   ├── data/                          # GitHub data ingestion
│   │   └── github_ingest.py
│   ├── preprocess/                    # Data cleaning & insight generation
│   │   ├── prepare_github_data.py
│   │   ├── finalize_dataset_github.py
│   │   ├── send_github_issues_to_llm.py
│   │   ├── summarize_github_issues.py
│   │   └── generate_business_tech_insights.py
│   ├── agent_pipeline/                # LangGraph agent workflows
│   │   ├── agent_github_ingest/       # Issue fetching agent
│   │   └── agent_embed_data/          # Embedding & Milvus insertion agent
│   ├── rag_pipeline/                  # RAG query & retrieval
│   │   ├── set_up_milvus_db.py
│   │   ├── form_data_for_collection_1.py
│   │   ├── form_data_for_collection_2.py
│   │   ├── generate_response.py
│   │   └── test_query.py
│   └── llm_finetuning/               # Phi-3 fine-tuning with LoRA
│       └── phi-instruct/
│           ├── model.py
│           ├── trainer.py
│           └── data.py
├── deployments/
│   └── deploy_rag_service/            # FastAPI deployment
│       ├── api_milvus.py
│       ├── start_service.sh
│       └── test_api.py
├── data/
│   └── processed/                     # Processed insights & embeddings

```
## Key Components

### 1. Data Ingestion

Fetches GitHub issues and comments via the GitHub API with rate-limit handling. Available as both a standalone script (`src/data/github_ingest.py`) and a LangGraph agent (`src/agent_pipeline/agent_github_ingest/`).

### 2. Preprocessing Pipeline

Cleans markdown, removes emojis, classifies issues (bug/feature/etc.), batches them, and uses an LLM (Ollama/Gemma) to extract **business insights** and **technical insights** per batch.

### 3. Vector Storage (Milvus)

Two collections:
- **issue_batches** -- batch-level metadata and concatenated summaries
- **issue_insights** -- individual insight embeddings (1024-dim BGE-m3 vectors) for semantic search

### 4. RAG Query Service

FastAPI endpoint that embeds user queries with BGE-m3, retrieves top-K similar insights from Milvus, and returns ranked results filtered by repo and insight type.

### 5. LLM Fine-tuning (Yet to be done)

Fine-tunes Microsoft Phi-3-mini-4k-instruct using 4-bit quantization + LoRA (r=8, alpha=16) with SFT to generate grounded answers from retrieved context.

## Setup

### Prerequisites

- Python 3.10
- GitHub personal access token
- Milvus (Lite or Server)

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Copy `.env_example` to `.env` and fill in:

```
GITHUB_TOKEN=''       # GitHub API token
MILVUS_URI=           # Milvus connection URI (default: localhost:19530)
DEVICE=               # cpu or cuda
```

## Usage

### Step 1: Ingest GitHub Issues

```bash
python src/agent_pipeline/agent_github_ingest/agent.py
```

### Step 2: Preprocess & Extract Insights

```bash
python src/preprocess/prepare_github_data.py
python src/preprocess/finalize_dataset_github.py
python src/preprocess/send_github_issues_to_llm.py
python src/preprocess/generate_business_tech_insights.py
```

### Step 3: Build Embeddings & Store in Milvus

```bash
python src/rag_pipeline/form_data_for_collection_1.py
python src/rag_pipeline/form_data_for_collection_2.py
python src/rag_pipeline/set_up_milvus_db.py
python src/agent_pipeline/agent_embed_data/agent.py
```

### Step 4: Deploy the API (Just to check if it is working)

```bash
cd deployments/deploy_rag_service
pip install -r requirements.txt
uvicorn api_milvus:app --reload
```

### Step 5: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are common dependency issues?", "top_k": 5}'
```

**API Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/query` | Search insights by semantic similarity |



## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph |
| Vector DB | Milvus |
| Embeddings | BAAI/bge-m3 (1024-dim) |
| Fine-tuning | LoRA + QLoRA (4-bit NF4) via PEFT/TRL |
| API | FastAPI + Uvicorn |
| Data Source | GitHub REST API |
