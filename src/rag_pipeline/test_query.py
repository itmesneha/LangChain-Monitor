from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pymilvus import MilvusClient

def get_client():
    return MilvusClient("./Repository_monitoring.db")


def load_issue_insights_collection(client):
    client.load_collection(collection_name="issue_insights")


def init_bge_embedder(device: str = "cpu"):
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": device, "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

def embed_query_bge(embedder: HuggingFaceBgeEmbeddings, query: str):
    return embedder.embed_query(query)


def search_insights(
    client,
    query: str,
    embedder: HuggingFaceBgeEmbeddings,
    top_k: int = 5,
    repo: str = "langchain",
    insight_type: str | None = None,  # "business" or "technical" or None
):
    qvec = embed_query_bge(embedder, query)

    # Build optional filter expression
    expr_parts = [f'repo == "{repo}"']
    if insight_type:
        expr_parts.append(f'insight_type == "{insight_type}"')
    expr = " && ".join(expr_parts)

    results = client.search(
        collection_name="issue_insights",
        data=[qvec],
        anns_field="insight_vector",
        limit=top_k,
        filter=expr,
        output_fields=[
            "insight_id",
            "batch_number",
            "insight_type",
            "insight_index",
            "insight_text",
        ],
    )

    # MilvusClient.search returns a list (one per query vector)
    hits = results[0]

    # Normalize output for printing / downstream use
    formatted = []
    for h in hits:
        # Depending on pymilvus version, fields might be in h["entity"] or directly in h
        entity = h.get("entity", h)
        formatted.append({
            "score": h.get("distance", h.get("score")),  # name varies by version
            "insight_id": entity.get("insight_id"),
            "batch_number": entity.get("batch_number"),
            "insight_type": entity.get("insight_type"),
            "insight_index": entity.get("insight_index"),
            "insight_text": entity.get("insight_text"),
        })

    return formatted


if __name__ == "__main__":
    client = get_client()
    load_issue_insights_collection(client)
    embedder = init_bge_embedder(device="cpu")

    query = "Why is upgrading to LangChain v1 failing because of dependency conflicts?"
    hits = search_insights(
        client=client,
        query=query,
        embedder=embedder,
        top_k=8,
        repo="langchain",
        insight_type="technical",   # try None too
    )

    for i, h in enumerate(hits, start=1):
        print(f"{i}. score={h['score']:.4f} | {h['insight_id']} | batch={h['batch_number']} | {h['insight_type']}")
        print(f"   {h['insight_text']}\n")
