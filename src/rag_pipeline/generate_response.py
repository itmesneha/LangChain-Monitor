import os
import json
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pymilvus import MilvusClient
from langchain.chat_models import init_chat_model

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

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


def load_llm_model(model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    return init_chat_model(
        model_name,
        model_provider="huggingface",
        temperature=0.2,
        max_tokens=100,
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )


from pydantic import BaseModel, Field
from typing import List


class Summary(BaseModel):
    overall_assessment: str = Field(
        ...,
        description="One concise sentence summarizing the overall technical state."
    )


class RecommendedStep(BaseModel):
    step: str = Field(
        ...,
        description="Short, actionable title for the recommendation."
    )
    details: str = Field(
        ...,
        description="Clear explanation of what to do and why."
    )


class LLMResponse(BaseModel):
    summary: Summary
    recommended_next_steps: List[RecommendedStep]




class RecommendedNextStepsResponse(BaseModel):
    recommended_next_steps: List[str] = Field(
        ...,
        description="List of clear, actionable recommended next steps."
    )


# json_schema = {
#     "title": "LLMResponse",
#     "type": "object",
#     "properties": {
#         "summary": {
#             "type": "object",
#             "properties": {
#                 "overall_assessment": {
#                     "type": "string",
#                     "description": "One concise sentence summarizing the overall technical state."
#                 }
#             },
#             "required": ["overall_assessment"],
#             "additionalProperties": False
#         },
#         "recommended_next_steps": {
#             "type": "array",
#             "items": {
#                 "type": "object",
#                 "properties": {
#                     "step": {
#                         "type": "string",
#                         "description": "Short, actionable title for the recommendation."
#                     },
#                     "details": {
#                         "type": "string",
#                         "description": "Clear explanation of what to do and why."
#                     }
#                 },
#                 "required": ["step", "details"],
#                 "additionalProperties": False
#             },
#             "minItems": 1
#         }
#     },
#     "required": ["summary", "recommended_next_steps"],
#     "additionalProperties": False
# }


def generate_content(llm, llm_input):

    model_with_structure = llm.with_structured_output(RecommendedNextStepsResponse, method="json_schema")

    system_content = '''
        You are a senior software engineer and technical advisor.
        Your task is to analyze the provided insights and produce a response
    '''
    example_user = '''
        {
        "repo": "payment-service",
        "insights": [
            "Database schema changes are frequently deployed without rollback plans.",
            "Memory usage spikes during peak traffic due to unbounded in-memory caching.",
            "Production incidents take longer to resolve because logs are inconsistent across services.",
            "Multiple services depend on outdated third-party libraries."
        ],
        "task": "Explain the issues and recommend practical next steps for the user."
        }
    '''
    example_assistant = '''
        {
        "recommended_next_steps": [
            "Require rollback plans for all database schema changes and validate them in deployment pipelines.",
            "Introduce cache eviction policies such as TTL or LRU to prevent unbounded memory growth during peak traffic.",
            "Standardize structured logging across services to improve observability and incident response times.",
            "Audit and upgrade outdated third-party dependencies to reduce security and stability risks."
        ]
        }
    '''
    conversation = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": example_assistant},
        {"role": "user", "content": json.dumps(llm_input)},
    ]
    # print("Conversation:", conversation)    
    response = model_with_structure.invoke(conversation, include_raw=False)

    return response


   


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

    llm = load_llm_model(model_name="Qwen/Qwen3-0.6B")
    response = generate_content(llm, llm_input)
    print(response)

