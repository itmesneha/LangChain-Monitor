import json
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
    limit_per_type: int = 3,
    top_k: int = 20,
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
    for i, h in enumerate(hits):
        # Depending on pymilvus version, fields might be in h["entity"] or directly in h
        if i == limit_per_type:
            break
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


def extract_insight_texts(hits, limit: int = 3):
    """
    Extract only insight_text strings for LLM context.
    """
    texts = []
    for h in hits[:limit]:
        text = h.get("insight_text", "").strip()
        if text:
            texts.append(text)
    return texts


def build_llm_context_text_only(
    client,
    embedder,
    query: str,
    repo: str,
    limit_per_type: int = 3,
):
    technical_hits = search_insights(
        client=client,
        query=query,
        embedder=embedder,
        limit_per_type=limit_per_type,
        repo=repo,
        insight_type="technical",
    )
    business_hits = search_insights(
        client=client,
        query=query,
        embedder=embedder,
        limit_per_type=limit_per_type,
        repo=repo,
        insight_type="business",
    )

    return {
        "query": query,
        "context": {
            "technical_insights": extract_insight_texts(technical_hits, limit_per_type),
            "business_insights": extract_insight_texts(business_hits, limit_per_type),
        }
    }

def build_llm_input(
    repo: str,
    technical_insights: list[str],
    business_insights: list[str] | None = None,
    task: str = "Explain the issues and recommend practical next steps for the user.",
):
    """
    Build the final JSON payload sent to the LLM.
    """
    insights = []

    # Combine insights (technical first, business next)
    if technical_insights:
        insights.extend(technical_insights)

    if business_insights:
        insights.extend(business_insights)

    return {
        "repo": repo,
        "insights": insights,
        "task": task,
    }


if __name__=="__main__":
    client = get_client()
    load_issue_insights_collection(client)
    embedder = init_bge_embedder(device="cpu")

    query = "which is the most pressing problem in langchain"
    context = build_llm_context_text_only(
        client, embedder, query, repo="langchain"
    )
    llm_input = build_llm_input(
        repo="langchain",
        technical_insights=context["context"]["technical_insights"],
        business_insights=context["context"]["business_insights"],
        task="Explain the issues and recommend practical next steps for the user.",
    )
    print(json.dumps(llm_input, indent=2))

