import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data/raw/github"
OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ISSUES_FILE = RAW_DATA_DIR / "langchain_issues.jsonl"
COMMENTS_FILE = RAW_DATA_DIR / "langchain_comments.jsonl"
OUTPUT_FILE = OUTPUT_DIR / "merged_github_issues.jsonl"

# ---------- LOAD DATA ----------
def load_json(path):
    # Support .jsonl (JSON Lines) files
    with open(path, "r", encoding="utf-8") as f:
        if str(path).endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)

issues = load_json(ISSUES_FILE)
comments = load_json(COMMENTS_FILE)

print(f"Loaded {len(issues)} issues and {len(comments)} comments.")

# ---------- GROUP COMMENTS BY ISSUE ----------
comments_by_issue = defaultdict(list)

for c in comments:
    # All fields are present and correctly typed in the JSONL
    issue_number = c["issue_number"]
    comments_by_issue[issue_number].append({
        "comment_id": c["comment_id"],
        "author": c["user"],
        "body": c["body"],
        "created_at": c["created_at"]
    })

print(f"Grouped comments for {len(comments_by_issue)} issues.")

# ---------- MERGE ----------
merged_records = []
for issue in issues:
    issue_number = issue["number"]
    merged = {
        "issue_number": issue_number,
        "title": issue["title"],
        "body": issue["body"],
        "labels": issue["labels"] if isinstance(issue["labels"], list) else [],
        "author": issue["user"],
        "created_at": issue["created_at"],
        "state": issue["state"],
        "comments": comments_by_issue.get(issue_number, []),
        "url": issue["url"]
    }
    merged_records.append(merged)

print(f"Merged {len(merged_records)} issue records.")

# ---------- SAVE ----------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for rec in merged_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ Saved merged data to {OUTPUT_FILE}")
