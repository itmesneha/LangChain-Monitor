import json

def form_issue_digest(batch_issues):
    """
    Turns batch_issues into a readable digest string.
    """
    lines = []
    for idx, issue in enumerate(batch_issues, start=1):
        category = issue.get("final_category", "unknown").upper()
        summary = issue.get("ollama_summary", "").strip()
        summary = summary.replace("\n", " ")
        lines.append(f"[{category}] {summary}")
    return "\n".join(lines)


def form_json_for_collection_1(batch_summary, batch_meta):
    data = []
    for idx, batch in enumerate(batch_summary):
        batch_num = batch["batch_num"]
        business_insights = json.dumps(batch["business_insights"], ensure_ascii=False)
        tech_insights = json.dumps(batch["technical_insights"], ensure_ascii=False)
        num_issues = batch["issue_count"]

        issue_digest = form_issue_digest(batch_meta[idx]["batch_issues"])
        repo = "langchain"

        data.append({
            "batch_id": f"langchain_batch_{batch_num}",
            "repo": repo,
            "batch_number": batch_num,
            "num_issues": num_issues,
            "issue_digest": issue_digest,
            "business_insights_json": business_insights,
            "technical_insights_json": tech_insights,
        })
    return data

    
if __name__ == "__main__":
    batch_summary = json.load(open("../../data/processed/insights_batch_10_summary.json", "r"))
    batch_meta = []
    with open("../../data/processed/insights_batch_10.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():          # skip empty lines
                record = json.loads(line)
                batch_meta.append(record)
    data = form_json_for_collection_1(batch_summary, batch_meta)
    with open("../../data/processed/collection_1.json", "w") as f:
        json.dump(data, f, indent=2)

