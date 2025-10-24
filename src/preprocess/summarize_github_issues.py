"""
Summarize GitHub issues using facebook/bart-large-cnn model.
Adds 'bart_summary' field to each record.
Rate limit: 100 requests per 5 minutes
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data/processed/final_github_data.jsonl"
OUTPUT_FILE = BASE_DIR / "data/processed/final_github_data_summarized.jsonl"

# Hugging Face API configuration
API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("Error: HF_TOKEN not found in .env file")
    exit(1)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
}

# Rate limiting: 100 requests per 5 minutes
RATE_LIMIT = 100  # requests
TIME_WINDOW = 300  # 5 minutes in seconds
request_times = []

print(f"Using model: facebook/bart-large-cnn")
print(f"Rate limit: {RATE_LIMIT} requests per {TIME_WINDOW // 60} minutes")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")


def check_and_wait_for_rate_limit():
    """Check rate limits and wait if necessary."""
    global request_times
    
    current_time = time.time()
    
    # Remove requests older than the time window
    request_times = [t for t in request_times if current_time - t < TIME_WINDOW]
    
    # Check if we've hit the rate limit
    if len(request_times) >= RATE_LIMIT:
        # Calculate how long to wait
        oldest_request = request_times[0]
        wait_time = TIME_WINDOW - (current_time - oldest_request) + 1
        print(f"\n⏳ Rate limit reached ({len(request_times)} requests in last 5 min). Waiting {wait_time:.0f}s...")
        time.sleep(wait_time)
        # Clean up old requests again after waiting
        current_time = time.time()
        request_times = [t for t in request_times if current_time - t < TIME_WINDOW]


def clean_text_for_bart(text):
    """Clean text to prevent tokenization issues."""
    if not text:
        return ''
    
    # Remove non-ASCII characters that might cause issues
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    # Remove very long words (likely corrupted data)
    words = text.split()
    words = [w for w in words if len(w) < 50]
    text = ' '.join(words)
    
    return text.strip()


def query_bart(text):
    """Query BART model for summarization with progressive fallback."""
    # Validate input text
    if not text or not text.strip():
        return ''
    
    # Clean text first
    text = clean_text_for_bart(text)
    
    # BART needs at least some minimum length to summarize
    if len(text.strip()) < 50:
        return text.strip()
    
    check_and_wait_for_rate_limit()
    
    # Try with progressively shorter text lengths
    max_lengths = [4000, 3000, 2000, 1500, 1000, 500]
    
    for max_len in max_lengths:
        current_text = text[:max_len] if len(text) > max_len else text
        
        try:
            payload = {
                "inputs": current_text,
                "parameters": {
                    "max_length": 130,
                    "min_length": 30,
                    "do_sample": False
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            request_times.append(time.time())
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('summary_text', '')
                return ''
            elif response.status_code == 400:
                # Try with shorter text
                if max_len > 500:
                    print(f"\n[WARNING] API 400 error at length {max_len}, trying shorter...")
                    continue
                else:
                    # Last resort - return truncated original
                    print(f"\n[WARNING] Failed even at 500 chars, using truncated text")
                    return current_text[:200]
            else:
                print(f"\n[ERROR] API returned status {response.status_code}: {response.text}")
                return ''
                
        except Exception as e:
            print(f"\n[ERROR] Request failed: {e}")
            if max_len > 500:
                continue  # Try shorter text
            return ''
    
    # If all attempts fail, return empty
    return ''

def clean_text_for_bart(text):
    """Clean text to prevent tokenization issues."""
    if not text:
        return ''
    
    # Remove non-ASCII characters that might cause issues
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    # Remove very long words (likely corrupted data)
    words = text.split()
    words = [w for w in words if len(w) < 50]
    text = ' '.join(words)
    
    return text.strip()

def query_bart(text):
    """Query BART model for summarization."""
    # Validate input text
    if not text or not text.strip():
        return ''
    
    # BART needs at least some minimum length to summarize
    # If text is too short, just return it as-is
    if len(text.strip()) < 50:
        return text.strip()
    
    check_and_wait_for_rate_limit()
    
    # Truncate text if too long (BART has 1024 token limit)
    # Roughly 4 chars per token, so limit to ~4000 chars to be safe
    if len(text) > 4000:
        text = text[:4000] + "..."
    
    try:
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": 130,
                "min_length": 30,
                "do_sample": False
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        request_times.append(time.time())
        
        if response.status_code == 200:
            result = response.json()
            # BART returns a list with one summary dict
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('summary_text', '')
            return ''
        elif response.status_code == 400:
            # Handle 400 errors (usually input issues) - return original text
            print(f"\n[WARNING] API 400 error for text (len={len(text)}): {response.text[:100]}")
            # For very short text after cleaning, just return it
            return text if len(text) < 200 else text[:200]
        else:
            print(f"\n[ERROR] API returned status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}")
        return None


def load_or_resume_records():
    """Load records, resuming from existing output file if it exists."""
    
    # Check if output file exists (resume scenario)
    if os.path.exists(OUTPUT_FILE):
        print(f"📂 Found existing output file: {OUTPUT_FILE}")
        print("Resuming from previous run...")
        records = []
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                records.append(record)
        
        already_summarized = sum(1 for r in records if "bart_summary" in r)
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
    records_to_summarize = [r for r in records if "bart_summary" not in r]
    print(f"Found {len(records_to_summarize)} records that need summarization")
    
    if is_resume:
        already_done = len([r for r in records if "bart_summary" in r])
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
        if "bart_summary" in record:
            continue
        
        # Prepare text for summarization
        title = record.get('title', '')
        body = record.get('body_clean', '')
        
        # Combine title and body
        input_text = f"{title}\n\n{body}" if body else title
        
        if not input_text.strip():
            record['bart_summary'] = ''
            continue
        
        # Get summary
        summary = query_bart(input_text)
        
        if summary is not None:
            record['bart_summary'] = summary
            summarized_count += 1
        else:
            record['bart_summary'] = ''  # Empty on error
        
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
    print(f"Total API requests: {len(request_times)}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
