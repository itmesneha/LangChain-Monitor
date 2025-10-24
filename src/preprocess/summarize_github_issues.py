"""
Summarize GitHub issues using local Ollama Gemma3:1b model.
Adds 'ollama_summary' field to each record.
No rate limits - runs locally!
"""

import json
import requests
from pathlib import Path
from tqdm import tqdm

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data/processed/final_github_data.jsonl"
OUTPUT_FILE = BASE_DIR / "data/processed/final_github_data_summarized.jsonl"

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"

print(f"Using model: {MODEL_NAME} (via Ollama)")
print(f"No rate limits - running locally!")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")


def query_ollama(text):
    """Query Ollama Gemma3:1b model for summarization."""
    # Validate input text
    if not text or not text.strip():
        return ''
    
    # If text is very short, just return it as-is
    if len(text.strip()) < 50:
        return text.strip()
    
    # Create a summarization prompt
    prompt = f"""Summarize the following GitHub issue in 2-3 concise sentences. Focus on the main problem, key details and any proposed solutions:

{text}

Summary:"""
    
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict":500  # Limit response length
            }
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get('response', '').strip()
            return summary
        else:
            print(f"\n[ERROR] Ollama API returned status {response.status_code}: {response.text}")
            return ''
            
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}")
        return ''


def load_or_resume_records():
    """Load records, resuming from existing output file if it exists."""
    
    # Check if output file exists (resume scenario)
    if OUTPUT_FILE.exists():
        print(f"📂 Found existing output file: {OUTPUT_FILE}")
        print("Resuming from previous run...")
        records = []
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                records.append(record)
        
        already_summarized = sum(1 for r in records if "ollama_summary" in r)
        print(f"Already summarized: {already_summarized} records")
        return records, True
    else:
        # Fresh start - load from input file
        print(f"📂 Starting fresh, loading from: {INPUT_FILE}")
        records = []
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                records.append(record)
        return records, False


def main():
    # Load records (resume if output file exists)
    print("Loading records...")
    records, is_resume = load_or_resume_records()
    print(f"Loaded {len(records)} total records")
    
    # Count records that need summarization
    records_to_summarize = [r for r in records if "ollama_summary" not in r]
    print(f"Found {len(records_to_summarize)} records that need summarization")
    
    if is_resume:
        already_done = len([r for r in records if "ollama_summary" in r])
        print(f"Already completed: {already_done} records")
    
    if len(records_to_summarize) == 0:
        print("\n✅ All records already summarized! Nothing to do.")
        return
    
    # Process records
    print("\n" + "="*80)
    print("✨ Generating summaries...")
    print("="*80)
    
    summarized_count = 0
    
    for i, record in enumerate(tqdm(records, desc="Summarizing issues")):
        # Skip if already has summary
        if "ollama_summary" in record:
            continue
        
        # Prepare text for summarization
        title = record.get('title', '')
        body = record.get('body_clean', '')
        
        # Combine title and body
        input_text = f"{title}\n\n{body}" if body else title
        
        if not input_text.strip():
            record['ollama_summary'] = ''
            continue
        
        # Get summary
        summary = query_ollama(input_text)
        
        if summary:
            record['ollama_summary'] = summary
            summarized_count += 1
        else:
            record['ollama_summary'] = ''  # Empty on error
        
        # Save after every 10 records
        if (summarized_count % 10 == 0 and summarized_count > 0) or summarized_count == len(records_to_summarize):
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Final save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Processing complete!")
    print(f"Total summarized in this run: {summarized_count}/{len(records_to_summarize)}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
