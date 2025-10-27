import json
import requests
import re
from typing import List, Dict
import time
from pathlib import Path

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"

# configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data/processed/final_github_data_summarized.jsonl"
OUTPUT_FILE = BASE_DIR / "data/processed/batch_github_data_business_tech_insights.jsonl"
BATCH_SIZES = [10, 20]  # Test different batch sizes [5,10,20]


def load_issues(file_path: str) -> List[Dict]:
    """Load issues from JSONL file"""
    issues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                issues.append(json.loads(line))
    return issues

def create_prompt(issues: List[Dict]) -> str:
    """Create a concise analysis prompt for batch issue insights."""
    prompt = (
        "You are an expert AI analyst. Analyze these GitHub issues and provide EXACTLY 10 insights.\n\n"
        "STRICT OUTPUT FORMAT (copy this exactly):\n"
        "BUSINESS INSIGHTS:\n"
        "1. [insight here]\n"
        "2. [insight here]\n"
        "3. [insight here]\n"
        "4. [insight here]\n"
        "5. [insight here]\n\n"
        "TECHNICAL INSIGHTS:\n"
        "1. [insight here]\n"
        "2. [insight here]\n"
        "3. [insight here]\n"
        "4. [insight here]\n"
        "5. [insight here]\n\n"
        "RULES:\n"
        "- Each insight must be ≤20 words\n"
        "- Start each line with the number (1-5)\n"
        "- Be specific and actionable\n"
        "- Focus on patterns across issues, not individual issues\n"
        "- NO extra text, explanations, or formatting\n\n"
        "===== ISSUES TO ANALYZE =====\n\n"
    )

    for i, issue in enumerate(issues, 1):
        prompt += f"Issue {i}:\n"
        prompt += f"Type: {issue.get('final_category', 'N/A')}\n"
        prompt += f"Summary: {issue.get('ollama_summary', 'N/A')}\n"

        comments = issue.get("comments", [])
        if comments:
            for comment in comments[:2]:  # limit to 2 comments
                text = comment.get("body_clean", comment.get("body", "")).strip().replace("\n", " ")
                if text:
                    prompt += f"Comment: {text[:150]}\n"
        prompt += "\n"

    prompt += (
        "===== END ISSUES =====\n\n"
        "Now provide your analysis in the EXACT format shown above:\n"
    )
    return prompt


def parse_insights(raw_text: str) -> Dict[str, List[str]]:
    """Parse and structure insights from LLM output"""
    
    parsed = {
        'business_insights': [],
        'technical_insights': []
    }
    
    # Remove markdown code blocks if present
    text = re.sub(r'```.*?```', '', raw_text, flags=re.DOTALL)
    text = re.sub(r'`', '', text)
    
    # Split into sections
    lines = text.split('\n')
    current_section = None
    
    # Placeholder patterns to ignore
    placeholder_patterns = [
        r'^\[insight here\]$',
        r'^\[.*?\]$',
        r'^\.\.\.+$',
        r'^insert\s+insight',
        r'^add\s+insight',
        r'^insight\s+\d+',
    ]
    
    def is_placeholder(text: str) -> bool:
        """Check if text is a placeholder"""
        text_lower = text.lower().strip()
        for pattern in placeholder_patterns:
            if re.match(pattern, text_lower):
                return True
        return False
    
    for line in lines:
        line = line.strip()
        
        # Detect section headers
        if re.search(r'business\s+insight', line, re.IGNORECASE):
            current_section = 'business'
            continue
        elif re.search(r'technical\s+insight', line, re.IGNORECASE):
            current_section = 'technical'
            continue
        
        # Extract numbered insights
        match = re.match(r'^(\d+)[\.\)\-\:]\s*(.+)$', line)
        if match and current_section:
            insight = match.group(2).strip()
            # Filter out placeholders, empty, or too-short insights
            if insight and len(insight) > 10 and not is_placeholder(insight):
                if current_section == 'business':
                    parsed['business_insights'].append(insight)
                elif current_section == 'technical':
                    parsed['technical_insights'].append(insight)
    
    # If parsing failed, try bullet points
    if not parsed['business_insights'] and not parsed['technical_insights']:
        current_section = None
        for line in lines:
            line = line.strip()
            if re.search(r'business', line, re.IGNORECASE):
                current_section = 'business'
                continue
            elif re.search(r'technical', line, re.IGNORECASE):
                current_section = 'technical'
                continue
            
            # Look for bullet points or dashes
            if line.startswith(('-', '•', '*')) and current_section:
                insight = line[1:].strip()
                # Filter out placeholders, empty, or too-short insights
                if insight and len(insight) > 10 and not is_placeholder(insight):
                    if current_section == 'business':
                        parsed['business_insights'].append(insight)
                    elif current_section == 'technical':
                        parsed['technical_insights'].append(insight)
    
    return parsed


def analyze_batch(issues_batch: List[Dict], model: str = MODEL_NAME) -> Dict:
    """Send batch to Ollama and get insights"""
    prompt = create_prompt(issues_batch)
    
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1000,
            }
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        raw_insights = result.get('response', '')
        
        # Parse the insights into structured format
        parsed_insights = parse_insights(raw_insights)
        
        return {
            'success': True,
            'business_insights': parsed_insights['business_insights'],
            'technical_insights': parsed_insights['technical_insights'],
            'raw_response': raw_insights,
            'issue_count': len(issues_batch),
            'model': model
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Request error: {str(e)}",
            'issue_count': len(issues_batch)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'issue_count': len(issues_batch)
        }

def process_batches(issues: List[Dict], batch_size: int, output_file: str, model: str = MODEL_NAME):
    """Process issues in batches and append results after each batch"""
    total_issues = len(issues)
    num_batches = (total_issues + batch_size - 1) // batch_size
    
    print(f"\n{'='*60}")
    print(f"Processing {total_issues} issues in batches of {batch_size}")
    print(f"Total batches: {num_batches}")
    print(f"Model: {model}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    results = []
    
    for i in range(0, total_issues, batch_size):
        batch = issues[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"\n--- Processing Batch {batch_num}/{num_batches} ---")
        print(f"Issues {i+1} to {min(i+batch_size, total_issues)}")
        
        start_time = time.time()
        result = analyze_batch(batch, model)
        elapsed = time.time() - start_time
        
        if result['success']:
            print(f"✓ Completed in {elapsed:.2f}s")
            
            # Display structured insights
            print("\n📊 BUSINESS INSIGHTS:")
            for idx, insight in enumerate(result.get('business_insights', []), 1):
                print(f"  {idx}. {insight}")
            
            print("\n🔧 TECHNICAL INSIGHTS:")
            for idx, insight in enumerate(result.get('technical_insights', []), 1):
                print(f"  {idx}. {insight}")
            
            print(f"\n✓ Found {len(result.get('business_insights', []))} business + {len(result.get('technical_insights', []))} technical insights")
        else:
            print(f"✗ Error: {result['error']}")
        
        batch_result = {
            'batch_num': batch_num,
            'batch_size': len(batch),
            'elapsed_time': elapsed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            **result
        }
        
        # Append result immediately after processing
        append_batch_result(batch_result, output_file, batch)
        results.append(batch_result)
        
        # Small delay between batches
        if batch_num < num_batches:
            time.sleep(1)
    
    return results

def save_results(results: List[Dict], output_file: str):
    """Save results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {output_file}")

def append_batch_result(result: Dict, output_file: str, issues_batch: List[Dict]):
    """Append a single batch result to JSONL file with all issues in the batch"""
    # Create one record per batch with all issues grouped together
    output_record = {
        'batch_issues': [
            {
                'ollama_summary': issue.get('ollama_summary', ''),
                'final_category': issue.get('final_category', ''),
                'comments': issue.get('comments', [])
            }
            for issue in issues_batch
        ],
        'business_insights': result.get('business_insights', []),
        'technical_insights': result.get('technical_insights', []),
        'raw_llm_response': result.get('raw_response', ''),
        'batch_number': result.get('batch_num', 0),
        'num_issues': len(issues_batch)
    }
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
    
    print(f"✓ Batch result appended to {output_file}")

def main():

    # Load issues
    print("Loading issues...")
    issues = load_issues(INPUT_FILE)
    print(f"Loaded {len(issues)} issues")
    
    # Process with different batch sizes
    for batch_size in BATCH_SIZES:
        print(f"\n{'#'*60}")
        print(f"TESTING BATCH SIZE: {batch_size}")
        print(f"{'#'*60}")
        
        # Create output file for this batch size
        batch_output_file = OUTPUT_FILE.parent / f"insights_batch_{batch_size}.jsonl"
        
        # Clear existing file if it exists
        if batch_output_file.exists():
            batch_output_file.unlink()
        
        results = process_batches(issues, batch_size, str(batch_output_file), MODEL_NAME)
        
        # Also save summary as JSON
        summary_file = OUTPUT_FILE.parent / f"insights_batch_{batch_size}_summary.json"
        save_results(results, str(summary_file))
        
        # Summary statistics
        successful = sum(1 for r in results if r['success'])
        total_time = sum(r['elapsed_time'] for r in results)
        avg_time = total_time / len(results) if results else 0
        
        print(f"\n--- Batch Size {batch_size} Summary ---")
        print(f"Total batches: {len(results)}")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per batch: {avg_time:.2f}s")
        
        time.sleep(2)  # Pause between batch size tests

if __name__ == "__main__":
    main()