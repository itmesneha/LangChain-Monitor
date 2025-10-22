
import os
import requests
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import time

# ---------- CONFIG ----------
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}"}
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SAVE_DIR = BASE_DIR / "data/raw/github"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------- LOAD DATA ----------
def fetch_issues(repo="langchain-ai/langchain", max_pages=30, state="all"):
    """
    Fetch issues & PRs (GitHub treats PRs as issues) from a repo.
    """
    print(f"Fetching issues from {repo}...")
    issues = []
    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {"state": state, "per_page": 100, "page": page}
        r = requests.get(url, headers=HEADERS, params=params)
        if r.status_code == 403:
            print("Rate limited. Sleeping for 60s...")
            time.sleep(60)
            continue
        if r.status_code != 200:
            print(f"Error {r.status_code}: {r.text}")
            break
        data = r.json()
        if not data:
            break
        for issue in data:
            if "pull_request" in issue:
                continue  # skip PRs if we want only issues
            issues.append({
                "id": issue["id"],
                "number": issue["number"],
                "title": issue["title"],
                "body": issue["body"],
                "labels": [l["name"] for l in issue["labels"]],
                "state": issue["state"],
                "created_at": issue["created_at"],
                "updated_at": issue["updated_at"],
                "user": issue["user"]["login"] if issue["user"] else None,
                "url": issue["html_url"],
                "comments_url": issue["comments_url"]
            })
        print(f"Page {page} → {len(issues)} total issues")
    return issues


def fetch_comments(issues):
    """
    Fetch comments for all issues.
    """
    comments = []
    for i, issue in enumerate(issues):
        url = issue["comments_url"]
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            continue
        for c in r.json():
            comments.append({
                "issue_number": issue["number"],
                "comment_id": c["id"],
                "user": c["user"]["login"] if c["user"] else None,
                "body": c["body"],
                "created_at": c["created_at"],
                "url": c["html_url"] if "html_url" in c else None
            })
        if i % 50 == 0:
            print(f"Fetched comments for {i} issues...")
        time.sleep(0.2)
    return comments

# ---------- SAVE DATA ----------
def save_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    repo = "langchain-ai/langchain"
    issues = fetch_issues(repo, max_pages=50)
    save_jsonl(SAVE_DIR / "langchain_issues.jsonl", issues)

    comments = fetch_comments(issues)
    save_jsonl(SAVE_DIR / "langchain_comments.jsonl", comments)

    print(f"Saved {len(issues)} issues and {len(comments)} comments.")
