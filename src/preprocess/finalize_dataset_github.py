import json
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data/processed/classified_github_data_relabeled.jsonl"
OUTPUT_FILE = BASE_DIR / "data/processed/final_github_data.jsonl"

# Read and process records
records = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        records.append(record)

print(f"Processing {len(records)} records...")

# Process each record
processed_records = []
for record in records:
    # Create new record with desired fields
    new_record = {}
    
    # Copy fields we want to keep (excluding emojis)
    for key in ['issue_number', 'title', 'body_clean', 'labels', 'created_at', 'state']:
        if key in record:
            new_record[key] = record[key]
    
    # Process comments: remove emojis field from each comment
    if 'comments' in record:
        cleaned_comments = []
        for comment in record['comments']:
            cleaned_comment = {
                'author': comment.get('author'),
                'created_at': comment.get('created_at'),
                'body_clean': comment.get('body_clean')
            }
            cleaned_comments.append(cleaned_comment)
        new_record['comments'] = cleaned_comments
    
    # Determine final_category: prefer category_llm if present, else use category
    if 'category_llm' in record and record['category_llm']:
        new_record['final_category'] = record['category_llm']
    elif 'category' in record:
        new_record['final_category'] = record['category']
    else:
        new_record['final_category'] = 'other'  # fallback
    
    processed_records.append(new_record)

# Save processed records
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for record in processed_records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"✅ Saved {len(processed_records)} processed records to {OUTPUT_FILE}")

# Show summary
from collections import Counter
category_counts = Counter(record['final_category'] for record in processed_records)
print("\n📊 Final Category Distribution:")
for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cat}: {count}")
