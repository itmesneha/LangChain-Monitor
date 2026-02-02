from states import EmbedInputState, EmbedWorkingState, EmbedOutputState
from nodes import initialize, build_collection_1, build_collection_2, init_milvus, upsert_to_milvus
from langgraph.graph import StateGraph, START, END

def build_graph():
    graph = StateGraph(EmbedWorkingState, input=EmbedInputState, output=EmbedOutputState)

    graph.add_node("initialize", initialize)
    graph.add_node("build_collection_1", build_collection_1)
    graph.add_node("build_collection_2", build_collection_2)
    graph.add_node("init_milvus", init_milvus)
    graph.add_node("upsert_to_milvus", upsert_to_milvus)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "build_collection_1")
    graph.add_edge("build_collection_1", "build_collection_2")
    graph.add_edge("build_collection_2", "init_milvus")
    graph.add_edge("init_milvus", "upsert_to_milvus")
    graph.add_edge("upsert_to_milvus", END)
    return graph

