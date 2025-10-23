import os
import json
import random
import requests
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

def needs_relabel(labels, category):
    """Check if an issue needs relabeling based on its current labels and category."""
    # Unlabeled issues
    if not labels or not isinstance(labels, list) or len(labels) == 0:
        return True
    
    # Multiple labels
    if len(labels) > 1:
        return True
    
    # Category is "other"
    if category == "other":
        return True
    
    return False

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data/processed/classified_github_data.jsonl"
CATEGORIES = ["bug", "feature", "question", "other"]

# Load Google AI Studio API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY2")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file")
    exit(1)

# Google AI Studio configuration
MODEL = "gemini-2.5-flash"
BATCH_SIZE = 3  # Process 3 issues per request
API_KEY = GOOGLE_API_KEY

# Rate limiting (Google AI Studio free tier)
# Gemini Flash has 15 RPM and 1500 RPD limits
RATE_LIMIT_RPM = 10  # Being conservative - 10 instead of 15
RATE_LIMIT_RPD = 1500  # Requests per day
request_count_minute = 0
request_count_day = 0
last_request_time = 0
day_start_time = time.time()

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # seconds

print(f"Using Google AI Studio model: {MODEL}")
print(f"Batch size: {BATCH_SIZE} issues per request")
print(f"Rate limits: {RATE_LIMIT_RPM} RPM, {RATE_LIMIT_RPD} RPD")
print(f"Max retries: {MAX_RETRIES} with exponential backoff")


def check_and_wait_for_rate_limit():
    """Check rate limits and wait if necessary."""
    global request_count_minute, request_count_day, last_request_time, day_start_time
    
    current_time = time.time()
    
    # Check if a day has passed (reset daily counter)
    if current_time - day_start_time >= 86400:  # 24 hours
        print("\n📅 24 hours passed, resetting daily counter")
        request_count_day = 0
        day_start_time = current_time
    
    # Check daily limit
    if request_count_day >= RATE_LIMIT_RPD:
        print(f"\n❌ Daily limit ({RATE_LIMIT_RPD}) reached. Cannot continue.")
        return False
    
    # Check minute limit and wait if needed
    time_since_last = current_time - last_request_time if last_request_time > 0 else 60
    
    if time_since_last < 60:  # Within the same minute
        if request_count_minute >= RATE_LIMIT_RPM:
            wait_time = 60 - time_since_last + 1
            print(f"⏳ Rate limit reached ({request_count_minute} requests in last minute). Waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
            request_count_minute = 0
            last_request_time = time.time()
    else:
        # More than a minute has passed, reset minute counter
        request_count_minute = 0
    
    return True

# Load all records and identify those needing relabeling
def load_records():
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    return records

def load_or_resume_records():
    """Load records, resuming from existing output file if it exists."""
    output_file = str(INPUT_FILE).replace(".jsonl", "_relabeled.jsonl")
    
    # Check if output file exists (resume scenario)
    if os.path.exists(output_file):
        print(f"📂 Found existing output file: {output_file}")
        print("Resuming from previous run...")
        records = []
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                records.append(record)
        
        already_labeled = sum(1 for r in records if "category_llm" in r)
        print(f"Already labeled: {already_labeled} records")
        return records, output_file, True
    else:
        # Fresh start - load from input file
        print(f"📂 Starting fresh, loading from: {INPUT_FILE}")
        records = load_records()
        return records, output_file, False

def evaluate_model_accuracy(records, sample_size=20):
    """Evaluate model accuracy on a random sample."""
    # Filter records that need relabeling
    eligible_records = [r for r in records if needs_relabel(r.get("labels", []), r.get("category", ""))]
    
    if not eligible_records:
        print("No records found that need relabeling.")
        return False
        
    # Take random sample
    sample_records = random.sample(eligible_records, min(sample_size, len(eligible_records)))
    correct = 0
    total = 0
    
    print(f"\n🔍 Evaluating model on {len(sample_records)} sample records...")
    
    for i, record in enumerate(sample_records, 1):
        current_category = record.get("category")
        llm_category = classify_issue_single(record["title"], record.get("body_clean", ""))
        
        if llm_category:
            total += 1
            # If current category is "unlabeled" and LLM gives any valid category, count as correct
            if current_category == "unlabeled" and llm_category in CATEGORIES:
                correct += 1
            elif llm_category == current_category:
                correct += 1
            print(f"[{i}/{len(sample_records)}] Title: {record['title'][:60]}...")
            print(f"   Current: {current_category}, LLM: {llm_category}, Match: {llm_category == current_category or current_category == 'unlabeled'}")
    
    if total == 0:
        print("No successful classifications in sample.")
        return False
        
    accuracy = correct / total
    print(f"\n📊 Model Accuracy: {accuracy:.2%} ({correct}/{total} correct)")
    return accuracy >= 0.8

def classify_issue_single(title, body):
    """Classify a single issue (used for spot-checking)."""
    result = classify_issues_batch([{"title": title, "body": body}])
    return result[0] if result else None

def classify_issues_batch(issues_batch):
    """Classify multiple issues in a single batch request with retry logic."""
    global request_count_minute, request_count_day, last_request_time
    
    # Check rate limits before making request
    if not check_and_wait_for_rate_limit():
        print("❌ Cannot proceed - rate limit exhausted")
        return [None] * len(issues_batch)
    
    # Build batch prompt
    batch_prompt = """You are labeling GitHub issues into one of four categories:
- bug: describes a problem or error in the system
- feature: requests a new feature or enhancement
- question: asks a clarification or usage question
- other: anything else (meta, docs, chore, etc.)

For each issue below, return ONLY the category name (bug, feature, question, or other) on a separate line.

"""
    
    for i, issue in enumerate(issues_batch, 1):
        batch_prompt += f"\n--- Issue {i} ---\n"
        batch_prompt += f"Title: {issue['title']}\n"
        batch_prompt += f"Body: {issue['body']}\n"
    
    batch_prompt += "\nRespond with one category per line (in order):"
    
    # Retry logic with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "contents": [{
                    "parts": [{
                        "text": batch_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 50000
                }
            }
            
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            
            # Update counters after request
            request_count_minute += 1
            request_count_day += 1
            last_request_time = time.time()
            
            if resp.status_code == 200:
                response_json = resp.json()
                print(response_json)
                
                try:
                    # Extract text from Google AI Studio response
                    llm_text = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                    
                    # Parse categories from response (one per line)
                    lines = [line.strip().lower() for line in llm_text.split('\n') if line.strip()]
                    categories = []
                    
                    for line in lines:
                        # Extract category from line
                        for category in CATEGORIES:
                            if category in line:
                                categories.append(category)
                                break
                    
                    # Ensure we have the right number of categories
                    while len(categories) < len(issues_batch):
                        categories.append("other")
                    
                    return categories[:len(issues_batch)]
                    
                except Exception as e:
                    print(f"Error parsing response: {e}")
                    if attempt < MAX_RETRIES - 1:
                        delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                        print(f"Retrying in {delay}s... (attempt {attempt + 2}/{MAX_RETRIES})")
                        time.sleep(delay)
                        continue
                    return [None] * len(issues_batch)
            
            elif resp.status_code == 429:
                # Rate limit hit
                print(f"\n⚠️ Rate limit error (429). Waiting 60s before retry...")
                time.sleep(60)  # Wait a full minute before retrying
                # Don't count this attempt's counters since request failed
                request_count_minute = max(0, request_count_minute - 1)
                request_count_day = max(0, request_count_day - 1)
                if attempt < MAX_RETRIES - 1:
                    continue
                return [None] * len(issues_batch)
            
            else:
                print(f"[ERROR] API returned status {resp.status_code}: {resp.text}")
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    print(f"Retrying in {delay}s... (attempt {attempt + 2}/{MAX_RETRIES})")
                    time.sleep(delay)
                    continue
                return [None] * len(issues_batch)
                
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                print(f"Retrying in {delay}s... (attempt {attempt + 2}/{MAX_RETRIES})")
                time.sleep(delay)
                continue
            return [None] * len(issues_batch)
    
    return [None] * len(issues_batch)

# Main execution
def main():
    # Load all records (resume if output file exists)
    print("Loading records...")
    records, output_file, is_resume = load_or_resume_records()
    print(f"Loaded {len(records)} total records")
    
    # Count records that need relabeling (excluding already labeled ones)
    def needs_relabel_check(r):
        # Skip if already has LLM label
        if "category_llm" in r:
            return False
        return needs_relabel(r.get("labels", []), r.get("category", ""))
    
    records_to_relabel = [r for r in records if needs_relabel_check(r)]
    print(f"Found {len(records_to_relabel)} records that need relabeling:")
    print(f"  - Unlabeled issues (no labels)")
    print(f"  - Multiple labels")
    print(f"  - Category is 'other'")
    if is_resume:
        already_done = len([r for r in records if "category_llm" in r])
        print(f"  - Already completed: {already_done} records")
    
    if len(records_to_relabel) == 0:
        print("\n✅ All records already labeled! Nothing to do.")
        return
    
    # Evaluate model accuracy (only if starting fresh)
    if not is_resume:
        print("\n" + "="*60)
        if not evaluate_model_accuracy(records):
            print("❌ Model accuracy below 80% threshold. Stopping.")
            return
    else:
        print("\n" + "="*60)
        print("⏭️  Skipping accuracy check (resuming from previous run)")
    
    # Process records that need relabeling
    print("\n" + "="*60)
    print("✨ Processing records...")
    print(f"Processing {len(records_to_relabel)} records in batches of {BATCH_SIZE}...")
    
    # Prepare batches of records to relabel (only those without category_llm)
    records_needing_relabel = [(i, r) for i, r in enumerate(records) if needs_relabel_check(r)]
    
    relabeled_count = 0
    batch_num = 0
    
    for batch_start in range(0, len(records_needing_relabel), BATCH_SIZE):
        batch = records_needing_relabel[batch_start:batch_start + BATCH_SIZE]
        batch_num += 1
        
        # Prepare batch for classification
        issues_batch = [{"title": r[1]["title"], "body": r[1].get("body_clean", "")} for r in batch]
        
        # Classify batch
        categories = classify_issues_batch(issues_batch)
        
        # Update records with classifications
        for (idx, _), category in zip(batch, categories):
            if category:
                records[idx]["category_llm"] = category
                relabeled_count += 1
        
        # Save entire dataset after each batch (overwrites file)
        with open(output_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        total_batches = (len(records_needing_relabel) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"✓ Batch {batch_num}/{total_batches} ({relabeled_count}/{len(records_to_relabel)} relabeled, {request_count_day} requests) → Saved")
    
    print(f"\n✅ Processing complete!")
    print(f"Total relabeled in this run: {relabeled_count}/{len(records_to_relabel)}")
    print(f"Total API requests: {request_count_day}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
