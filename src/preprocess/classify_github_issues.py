# labels are the input - they are the original GitHub labels that come with each issue
# category is determined by the classify_issue() function which:
# Takes the issue's labels as input
# Converts them all to lowercase
# Uses simple rule-based logic to map the labels to one of four categories:
# "bug": if any label contains "bug", "error", or "fix"
# "feature": if any label contains "feature", "enhancement", or "improvement"
# "question": if any label contains "question", "help", or "support"
# "other": as a default fallback if none of the above match
# "unlabeled": if the issue has no labels


import json
from pathlib import Path
from tqdm import tqdm

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data/processed/clean_github_data.jsonl"
OUTPUT_FILE = BASE_DIR / "data/processed/classified_github_data.jsonl"

# ---------- Simple rule-based classifier ----------
def classify_issue(labels):
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


def main():
    unlabeled_count = 0
    total_issues = 0
    labeled_counts = {"bug": 0, "feature": 0, "question": 0, "other": 0}
    multi_label_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Classifying issues"):
            issue = json.loads(line)
            total_issues += 1

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
            fout.write(json.dumps(issue, ensure_ascii=False) + "\n")

    print("\n✅ Classification summary:")
    print(f"Total issues: {total_issues}")
    print(f"Unlabeled issues: {unlabeled_count} ({unlabeled_count / total_issues:.2%})\n")
    print(f"Issues with multiple labels: {multi_label_count} ({multi_label_count / total_issues:.2%})")
    print("Label distribution:")
    for cat, count in labeled_counts.items():
        print(f"  {cat:10s}: {count}")

    print(f"\nSaved classified data → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
