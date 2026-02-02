import json
from pymilvus import MilvusClient
from states import EmbedInputState, EmbedWorkingState, EmbedOutputState
import sys
sys.path.append("../..")
from rag_pipeline.form_data_for_collection_1 import form_json_for_collection_1
from rag_pipeline.form_data_for_collection_2 import build_rows_from_batches
from rag_pipeline.set_up_milvus_db import create_database, create_schema_for_collection_1, create_collection_1,\
      create_schema_for_collection_2, create_collection_2, insert_to_collection_1, insert_to_collection_2, create_index_for_vectors


def initialize(state: EmbedInputState) -> EmbedWorkingState:
    with open(state["batch_summary_path"], "r") as f:
        batch_summaries = json.load(f)

    batch_meta = []
    with open(state["batch_meta_path"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                batch_meta.append(json.loads(line))

    return {
        **state,
        "batch_summaries": batch_summaries,
        "batch_meta": batch_meta,
        "collection_1_rows": [],
        "collection_2_rows": [],
        "milvus_client": None,
        "error": None,
    }


def build_collection_1(state: EmbedWorkingState):
    data = form_json_for_collection_1(
        state["batch_summaries"],
        state["batch_meta"],
    )
    state["collection_1_rows"] = data
    return state


def build_collection_2(state: EmbedWorkingState):
    rows = build_rows_from_batches(
        state["batch_summaries"],
        repo=state["repo"],
    )
    state["collection_2_rows"] = rows
    return state


def init_milvus(state: EmbedWorkingState):
    # Default to your current Milvus Lite file usage
    uri = state.get("milvus_uri") or "./Repository_monitoring.db"
    client = MilvusClient(uri)

    db_name = state.get("milvus_db_name")

    # If user provided a DB name, treat it as Milvus Server style (or server-like)
    if db_name:
        # Some Milvus deployments support DB APIs; MilvusClient exposes these in recent versions.
        # We'll be defensive: try-create, ignore if not supported / already exists.
        try:
            existing = client.list_databases()
            if db_name not in existing:
                client.create_database(db_name=db_name)
            client.use_database(db_name=db_name)
        except Exception:
            # If DB APIs aren't available (Milvus Lite / older server), we proceed without named DB.
            pass

    state["milvus_client"] = client
    return state

def upsert_to_milvus(state: EmbedWorkingState) -> EmbedOutputState:
    client = state["milvus_client"]

    # ---------- Collection 1: issue_batches ----------
    if not client.has_collection("issue_batches"):
        schema_1 = create_schema_for_collection_1()
        create_collection_1(client, schema_1)

    insert_to_collection_1(
        client,
        "issue_batches",
        state["collection_1_rows"],
    )

    # ---------- Collection 2: issue_insights ----------
    if not client.has_collection("issue_insights"):
        schema_2 = create_schema_for_collection_2()
        create_collection_2(client, schema_2)

    # Ensure index exists (idempotent)
    try:
        indexes = client.list_indexes(collection_name="issue_insights")
        index_names = {ix.get("index_name") for ix in indexes} if indexes else set()
        if "vector_index" not in index_names:
            create_index_for_vectors(client)
    except Exception:
        # Some deployments may not support list_indexes in the same way; safe fallback:
        # try creating index and ignore failure if it already exists.
        try:
            create_index_for_vectors(client)
        except Exception:
            pass

    insert_to_collection_2(
        client,
        "issue_insights",
        state["collection_2_rows"],
    )

    db_used = state.get("milvus_db_name") or (state.get("milvus_uri") or "./Repository_monitoring.db")

    return {
        "inserted_batches": len(state["collection_1_rows"]),
        "inserted_insights": len(state["collection_2_rows"]),
        "database_used": db_used,
    }
