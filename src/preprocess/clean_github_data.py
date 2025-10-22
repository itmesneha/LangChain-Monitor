import json
import re
from pathlib import Path
from tqdm import tqdm

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data/processed/merged_github_issues.jsonl"
OUTPUT_FILE = BASE_DIR / "data/processed/clean_github_data.jsonl"

# ---------- TEXT CLEANING HELPERS ----------

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
    if not text or not isinstance(text, str):
        return []
    return emoji_pattern.findall(text)

# Remove markdown formatting, code, and URLs
def clean_markdown(text):
    if not text or not isinstance(text, str):
        return ""

    # Extract and remove code blocks (```python ...```)
    # code_snippets = re.findall(r"```(?:.|\n)*?```", text)
    text = re.sub(r"```(?:.|\n)*?```", " [code] ", text)

    # Remove inline code (`code`)
    text = re.sub(r"`[^`]+`", " [code] ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " [link] ", text)

    # Remove markdown formatting symbols (*, _, #, >, -)
    text = re.sub(r"[*_>#`-]{1,}", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
    # code_snippets


def clean_comments(comment_list):
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


# ---------- MAIN PIPELINE ----------
def main():
    output_file = open(OUTPUT_FILE, "w", encoding="utf-8")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Cleaning GitHub data"):
            issue = json.loads(line)


            # Clean main body and extract emojis
            body = issue.get("body", "")
            body_clean = clean_markdown(body)
            emojis = extract_emojis(body)

            # Clean comments (with emoji extraction)
            comments_clean = clean_comments(issue.get("comments", []))

            # Skip empty issues
            if not body_clean and not comments_clean:
                continue

            cleaned_issue = {
                "issue_number": issue.get("issue_number"),
                "title": issue.get("title", "").strip(),
                "body_clean": body_clean,
                "emojis": emojis,
                # "code_snippets": code_snippets,
                "labels": issue.get("labels", []),
                "author": issue.get("author"),
                "created_at": issue.get("created_at"),
                "state": issue.get("state"),
                "comments": comments_clean,
                "url": issue.get("url")
            }

            output_file.write(json.dumps(cleaned_issue, ensure_ascii=False) + "\n")

    output_file.close()
    print(f"✅ Saved cleaned data to {OUTPUT_FILE}")

# ---------- ENTRY ----------
if __name__ == "__main__":
    main()
