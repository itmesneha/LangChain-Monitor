from graph import build_graph
from states import GitHubOutputState

if __name__=="__main__":
    graph_instance = build_graph()
    graph = graph_instance.compile()
    result: GitHubOutputState = graph.invoke({
        "repo": "langchain-ai/langchain",
        "max_pages": 2,
        "state": "all",
    })
