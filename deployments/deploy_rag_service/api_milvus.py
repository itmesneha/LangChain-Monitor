from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from pymilvus import MilvusClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os

# ---------------
# CONFIG
# ---------------
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = "issue_insights"
VECTOR_FIELD = "insight_vector"
DEVICE = os.getenv("DEVICE", "cpu")

# ---------------
# MODELS
# ---------------
class QueryRequest(BaseModel):
    query: str
    repo: str = "langchain"
    top_k: int = Field(default=5, ge=1, le=50)
    insight_type: Optional[Literal["business", "technical"]] = None

class InsightHit(BaseModel):
    insight_id: str
    repo: str
    batch_number: int
    insight_type: str
    insight_index: int
    insight_text: str
    score: float

class QueryResponse(BaseModel):
    query: str
    top_k: int
    results: List[InsightHit]

# ---------------
# SETUP
# ---------------
app = FastAPI(title="Milvus RAG API (Server Mode)")

client = MilvusClient("../../src/rag_pipeline/Repository_monitoring.db")
embedder = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": DEVICE, "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)

@app.on_event("startup")
def load():
    client.load_collection(COLLECTION)

@app.on_event("shutdown")
def shutdown():
    client.close()

# ---------------
# API
# ---------------
@app.get("/health")
def health():
    return {"status": "ok", "milvus_uri": MILVUS_URI}

@app.post("/query", response_model=QueryResponse)
def query_vector(req: QueryRequest):
    vec = embedder.embed_query(req.query)

    filters = [f'repo == "{req.repo}"']
    if req.insight_type:
        filters.append(f'insight_type == "{req.insight_type}"')
    expr = " && ".join(filters)

    raw = client.search(
        collection_name=COLLECTION,
        data=[vec],
        anns_field=VECTOR_FIELD,
        limit=req.top_k,
        filter=expr,
        output_fields=[
            "insight_id", "repo", "batch_number", "insight_type",
            "insight_index", "insight_text"
        ]
    )

    results = []
    for hit in raw[0]:
        entity = hit.get("entity", hit)
        results.append(InsightHit(
            insight_id=entity["insight_id"],
            repo=entity["repo"],
            batch_number=entity["batch_number"],
            insight_type=entity["insight_type"],
            insight_index=entity["insight_index"],
            insight_text=entity["insight_text"],
            score=float(hit.get("score", hit.get("distance", 0)))
        ))

    return QueryResponse(query=req.query, top_k=req.top_k, results=results)
