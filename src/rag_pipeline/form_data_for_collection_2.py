
'''
insight_id	VARCHAR (PK)	"langchain_b1_tech_03"
repo	VARCHAR	"langchain"
batch_number	INT64	Links to parent batch
insight_type	VARCHAR	"business" or "technical"
insight_index	INT64	Order within batch (1–N)
insight_text	VARCHAR	The actual insight bullet
Vector field (required)
embedding	FLOAT_VECTOR	Embedding of insight_text
'''

from typing import Any, Dict, List, Callable, Optional, Tuple
import time
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def make_insight_id(repo: str, batch_num: int, insight_type: str, insight_index: int) -> str:
    """
    Creates IDs like: langchain_b1_tech_03
    - repo: "langchain"
    - batch_num: 1
    - insight_type: "technical" or "business"
    - insight_index: 1..N

    Output examples:
      - langchain_b1_tech_03
      - langchain_b1_business_01
    """
    suffix = "tech" if insight_type == "technical" else "business"
    return f"{repo}_b{batch_num}_{suffix}_{insight_index:02d}"




def init_bge_embedder(device: str = "cpu"):
    model_name = "BAAI/bge-m3"
    model_kwargs = {
        "device": device,
        "trust_remote_code": True
    }
    encode_kwargs = {
        "normalize_embeddings": True
    }

    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def embed_text_bge(
    embedder: HuggingFaceBgeEmbeddings,
    texts: str,
) -> List[float]:
    """
    Embeds a list of texts using BGE.
    Returns List[List[float]] suitable for Milvus FLOAT_VECTOR.
    """
    if not texts:
        return []

    # LangChain BGE supports embed_documents for batching
    embeddings = embedder.embed_query(texts)
    return embeddings

def build_rows_from_one_batch(
    embedder,
    batch_json: Dict[str, Any],
    repo: str = "langchain",
) -> List[Dict[str, Any]]:
    """
    Builds Milvus rows for one batch JSON, excluding embeddings for now.

    Expected keys:
      - batch_num (int)
      - business_insights (list[str])
      - technical_insights (list[str])
    """
    batch_num = batch_json.get("batch_num")
    if batch_num is None:
        raise ValueError("batch_json missing required key: 'batch_num'")

    rows: List[Dict[str, Any]] = []

    for i, text in enumerate(batch_json.get("business_insights", []), start=1):
        txt_emb = embed_text_bge(embedder, text)
        rows.append({
            "insight_id": make_insight_id(repo, int(batch_num), "business", i),
            "repo": repo,
            "batch_number": int(batch_num),
            "insight_type": "business",
            "insight_index": int(i),
            "insight_text": (text or "").strip(),
            "insight_vector": txt_emb
        })

    for i, text in enumerate(batch_json.get("technical_insights", []), start=1):
        txt_emb = embed_text_bge(embedder, text)
        rows.append({
            "insight_id": make_insight_id(repo, int(batch_num), "technical", i),
            "repo": repo,
            "batch_number": int(batch_num),
            "insight_type": "technical",
            "insight_index": int(i),
            "insight_text": (text or "").strip(),
            "insight_vector": txt_emb
        })

    return rows


def build_rows_from_batches(
    batches: List[Dict[str, Any]],
    repo: str = "langchain",
) -> List[Dict[str, Any]]:
    """
    Builds Milvus rows for many batch JSON objects, excluding embeddings for now.
    """
    embedder = init_bge_embedder()
    all_rows: List[Dict[str, Any]] = []
    for b in batches:
        all_rows.extend(build_rows_from_one_batch(embedder, b, repo=repo))
    return all_rows


if __name__ == "__main__":
    import json
    from pprint import pprint

    with open("../../data/processed/insights_batch_10_summary.json", "r") as f:
        batches = json.load(f)

    rows = build_rows_from_batches(batches)

    with open("../../data/processed/collection_2.json", "w") as f:
        json.dump(rows, f, indent=2)
   