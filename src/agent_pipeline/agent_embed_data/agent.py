from graph import build_graph
from states import EmbedOutputState

if __name__=="__main__":
    graph_instance = build_graph()
    graph = graph_instance.compile()
    result: EmbedOutputState = graph.invoke({
        "repo": "langchain",
        "batch_summary_path": "../../../data/processed/insights_batch_10_summary.json",
        "batch_meta_path": "../../../data/processed/insights_batch_10.jsonl",
        "milvus_uri": "./Repository_monitoring.db",
    })
