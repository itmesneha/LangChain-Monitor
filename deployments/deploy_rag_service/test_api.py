import requests

query = {
    "query": "dependency issue with langgraph supervisor",
    "repo": "langchain",
    "top_k": 5,
    "insight_type": "technical"
}

resp = requests.post("http://localhost:8000/query", json=query)
print(resp.status_code)
print(resp.json())
