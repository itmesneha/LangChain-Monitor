from langgraph.graph import StateGraph, START, END
from states import GitHubWorkingState, GitHubInputState, GitHubOutputState
from nodes import initialize, fetch_issues, next_issue, fetch_comments, assemble_issue, finalize_output

def build_graph():
    graph = StateGraph(GitHubWorkingState, input=GitHubInputState, output=GitHubOutputState)

    graph.add_node("initialize", initialize)
    graph.add_node("fetch_issues", fetch_issues)
    graph.add_node("next_issue", next_issue)
    graph.add_node("fetch_comments", fetch_comments)
    graph.add_node("assemble_issue", assemble_issue)
    graph.add_node("finalize_output", finalize_output)

    graph.add_edge(START, "initialize")

    graph.add_edge("initialize", "fetch_issues")
    graph.add_edge("fetch_issues", "next_issue")
    graph.add_edge("next_issue", "fetch_comments")
    graph.add_edge("fetch_comments", "assemble_issue")

    def should_continue(state):
        return "loop" if state["current_issue"] else "done"

    graph.add_conditional_edges(
        "assemble_issue",
        should_continue,
        {
            "loop": "next_issue",
            "done": "finalize_output",
        }
    )

    graph.add_edge("finalize_output", END)

    return graph
