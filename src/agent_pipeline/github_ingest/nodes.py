import os
import time
import requests
from dotenv import load_dotenv
import json
from pathlib import Path
from states import GitHubInputState, GitHubWorkingState, GitHubOutputState
load_dotenv()

HEADERS = {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}


def initialize(state: GitHubInputState) -> GitHubWorkingState:
    return GitHubWorkingState(
        repo=state.get("repo", ""),
        max_pages=state.get("max_pages", 10),
        state=state.get("state", "all"),

        issues=[],
        current_index=0,
        current_issue=None,
        current_comments=None,

        output=[],
    )


def fetch_issues(state: GitHubWorkingState):
    issues = []
    repo = state["repo"]

    for page in range(1, state["max_pages"] + 1):
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {"state": state["state"], "per_page": 100, "page": page}

        r = requests.get(url, headers=HEADERS, params=params)

        if r.status_code == 403:
            time.sleep(60)
            continue

        if r.status_code != 200:
            break

        data = r.json()
        if not data:
            break

        for issue in data:
            if "pull_request" in issue:
                continue

            issues.append({
                "number": issue["number"],
                "title": issue["title"],
                "body": issue["body"] or "",
                "labels": [l["name"] for l in issue["labels"]],
                "state": issue["state"],
                "created_at": issue["created_at"],
                "updated_at": issue["updated_at"],
                "user": issue["user"]["login"] if issue["user"] else None,
                "url": issue["html_url"],
                "comments_url": issue["comments_url"],
            })

    state["issues"] = issues
    state["current_index"] = 0
    return state


def next_issue(state: GitHubWorkingState):
    idx = state["current_index"]

    if idx >= len(state["issues"]):
        state["current_issue"] = None
        return state

    state["current_issue"] = state["issues"][idx]
    state["current_index"] += 1
    return state


def fetch_comments(state: GitHubWorkingState):
    issue = state["current_issue"]

    if not issue:
        state["current_comments"] = []
        return state

    r = requests.get(issue["comments_url"], headers=HEADERS)
    if r.status_code != 200:
        state["current_comments"] = []
        return state

    state["current_comments"] = [{
        "comment_id": c["id"],
        "user": c["user"]["login"] if c["user"] else None,
        "body": c["body"] or "",
        "created_at": c["created_at"],
        "url": c.get("html_url"),
    } for c in r.json()]

    return state



def assemble_issue(state: GitHubWorkingState):
    issue = state["current_issue"]
    comments = state["current_comments"] or []

    if not issue:
        return state

    enriched = {
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"],
        "labels": issue["labels"],
        "state": issue["state"],
        "created_at": issue["created_at"],
        "updated_at": issue["updated_at"],
        "user": issue["user"],
        "url": issue["url"],
        "comments": comments,
    }

    state["output"].append(enriched)
    return state



def write_jsonl(path: str | Path, items: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def save_jsonl(state: GitHubWorkingState) -> GitHubOutputState:
    output_path = "data/github_issues.jsonl"

    write_jsonl(output_path, state["output"])

    return {
        "issues": state["output"]
    }