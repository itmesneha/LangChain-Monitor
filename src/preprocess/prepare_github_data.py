"""
Combined preprocessing pipeline for GitHub data.
This script combines:
1. merge_github_data.py - Merges issues and comments
2. clean_github_data.py - Cleans text, removes markdown, extracts emojis
3. classify_github_issues.py - Classifies issues using rule-based logic

Run this after: src/data/github_ingest.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data/raw/github"
OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ISSUES_FILE = RAW_DATA_DIR / "langchain_issues.jsonl"
COMMENTS_FILE = RAW_DATA_DIR / "langchain_comments.jsonl"
FINAL_OUTPUT_FILE = OUTPUT_DIR / "classified_github_data.jsonl"

# ---------- STEP 1: LOAD & MERGE ----------
print("=" * 80)
print("STEP 1: Loading and merging issues with comments")
print("=" * 80)

def load_jsonl(path):
    """Load JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

issues = load_jsonl(ISSUES_FILE)
comments = load_jsonl(COMMENTS_FILE)

print(f"Loaded {len(issues)} issues and {len(comments)} comments.")

# Group comments by issue
comments_by_issue = defaultdict(list)
for c in comments:
    issue_number = c["issue_number"]
    comments_by_issue[issue_number].append({
        "comment_id": c["comment_id"],
        "author": c["user"],
        "body": c["body"],
        "created_at": c["created_at"]
    })

print(f"Grouped comments for {len(comments_by_issue)} issues.")

# Merge issues with their comments
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

# ---------- STEP 2: CLEAN TEXT ----------
print("\n" + "=" * 80)
print("STEP 2: Cleaning markdown, extracting emojis")
print("=" * 80)

# Emoji pattern
emoji_pattern = re.compile(
    r"[\U0001F600-\U0001F64F"  # emoticons
    r"\U0001F300-\U0001F5FF"  # symbols & pictographs
    r"\U0001F680-\U0001F6FF"  # transport & map symbols
    r"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    r"\U00002700-\U000027BF"  # Dingbats
    r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    r"\U00002600-\U000026FF"  # Misc symbols
    r"\U00002B50-\U00002B55"  # Stars
    r"]+",
    flags=re.UNICODE
)

def extract_emojis(text):
    """Extract emojis from text."""
    if not text or not isinstance(text, str):
        return []
    return emoji_pattern.findall(text)

def clean_markdown(text):
    """Remove markdown formatting, code, and URLs."""
    if not text or not isinstance(text, str):
        return ""

    # Remove code blocks
    text = re.sub(r"```(?:.|\n)*?```", " [code] ", text)
    # Remove inline code
    text = re.sub(r"`[^`]+`", " [code] ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " [link] ", text)
    # Remove markdown symbols
    text = re.sub(r"[*_>#`-]{1,}", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def clean_comments(comment_list):
    """Clean comment bodies and extract emojis."""
    cleaned_comments = []
    for c in comment_list:
        body = c.get("body", "")
        body_clean = clean_markdown(body)
        emojis = extract_emojis(body)
        if body_clean.strip():
            cleaned_comments.append({
                "author": c.get("author"),
                "created_at": c.get("created_at"),
                "body_clean": body_clean,
                "emojis": emojis
            })
    return cleaned_comments

# Clean all records
cleaned_records = []
for issue in tqdm(merged_records, desc="Cleaning data"):
    body = issue.get("body", "")
    body_clean = clean_markdown(body)
    emojis = extract_emojis(body)
    comments_clean = clean_comments(issue.get("comments", []))

    # Skip empty issues
    if not body_clean and not comments_clean:
        continue

    cleaned_issue = {
        "issue_number": issue.get("issue_number"),
        "title": issue.get("title", "").strip(),
        "body_clean": body_clean,
        "emojis": emojis,
        "labels": issue.get("labels", []),
        "author": issue.get("author"),
        "created_at": issue.get("created_at"),
        "state": issue.get("state"),
        "comments": comments_clean,
        "url": issue.get("url")
    }
    cleaned_records.append(cleaned_issue)

print(f"Cleaned {len(cleaned_records)} issue records (skipped {len(merged_records) - len(cleaned_records)} empty).")

# ---------- STEP 3: CLASSIFY ISSUES ----------
print("\n" + "=" * 80)
print("STEP 3: Classifying issues using rule-based logic")
print("=" * 80)

def classify_issue(labels):
    """Classify issue based on labels using simple heuristics."""
    if not labels:
        return None  # will count as unlabeled

    labels_lower = [l.lower() for l in labels]

    # Heuristics
    if any("bug" in l or "error" in l or "fix" in l for l in labels_lower):
        return "bug"
    if any("feature" in l or "enhancement" in l or "improvement" in l for l in labels_lower):
        return "feature"
    if any("question" in l or "help" in l or "support" in l for l in labels_lower):
        return "question"

    # Default fallback
    return "other"

# Classify all records
unlabeled_count = 0
labeled_counts = {"bug": 0, "feature": 0, "question": 0, "other": 0}
multi_label_count = 0

classified_records = []
for issue in tqdm(cleaned_records, desc="Classifying issues"):
    labels = issue.get("labels", [])
    
    if isinstance(labels, list) and len(labels) > 1:
        multi_label_count += 1

    category = classify_issue(labels)
    if category is None:
        unlabeled_count += 1
        category = "unlabeled"
    else:
        labeled_counts[category] += 1

    issue["category"] = category
    classified_records.append(issue)

# ---------- SAVE FINAL OUTPUT ----------
print("\n" + "=" * 80)
print("SAVING FINAL OUTPUT")
print("=" * 80)

with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in classified_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(classified_records)} classified records to {FINAL_OUTPUT_FILE}")

# ---------- SUMMARY ----------
print("\n" + "=" * 80)
print("CLASSIFICATION SUMMARY")
print("=" * 80)
print(f"Total issues: {len(classified_records)}")
print(f"Unlabeled issues: {unlabeled_count} ({unlabeled_count / len(classified_records):.2%})")
print(f"Issues with multiple labels: {multi_label_count} ({multi_label_count / len(classified_records):.2%})")
print("\nLabel distribution:")
for cat, count in labeled_counts.items():
    print(f"  {cat:10s}: {count}")
