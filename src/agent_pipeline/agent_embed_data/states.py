from typing import TypedDict, List, Dict, Any, Optional

class EmbedInputState(TypedDict):
    repo: str
    batch_summary_path: str
    batch_meta_path: str

    # Optional: Milvus Lite (file path)
    milvus_uri: Optional[str]          # e.g. "./Repository_monitoring.db"

    # Optional: Milvus Server database
    milvus_db_name: Optional[str]      # e.g. "Repository_monitoring"


class EmbedWorkingState(EmbedInputState):
    batch_summaries: List[Dict[str, Any]]
    batch_meta: List[Dict[str, Any]]
    collection_1_rows: List[Dict[str, Any]]
    collection_2_rows: List[Dict[str, Any]]
    milvus_client: Any
    error: Optional[str]


class EmbedOutputState(TypedDict):
    inserted_batches: int
    inserted_insights: int
    database_used: str
