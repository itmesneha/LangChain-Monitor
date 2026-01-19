import json
import random

'''
SYSTEM
You are a senior software engineer and technical advisor.

Your goal is to help a developer understand technical issues
and suggest clear, practical next steps.

Rules:
- Use ONLY the provided insights
- Do not invent issues
- Be concise, specific, and actionable
- Prioritize the most impactful recommendations
- If the insights are insufficient, say so
- Output valid JSON only


USER
{
  "repo": "my-repo",
  "insights": [
    "Database migrations are failing due to missing rollback scripts."
    "High memory usage caused by unbounded in-memory caching."
  ],
  "task": "Explain the issues and recommend practical next steps for the user."
}

ASSISTANT
{
  "summary": {
    "overall_assessment": "The repository contains a small number of technical issues that may impact reliability and performance if left unaddressed."
  },
  "recommended_next_steps": [
    {
      "step": "Require rollback scripts for all database migrations",
      "details": "Ensure every migration includes a tested rollback and add validation checks in CI pipelines.",
    },
    {
      "step": "Add cache limits and eviction policies",
      "details": "Introduce TTL or LRU eviction for caches and monitor memory usage in production.",
    }
  ],
}


'''


SYSTEM_PROMPT = (
    "You are a senior software engineer and technical and business advisor.\n\n"
    "Your goal is to help a developer understand technical issues and business issues\n"
    "and suggest clear, practical next steps.\n\n"
    "Rules:\n"
    "- Use ONLY the provided insights\n"
    "- Do not invent issues\n"
    "- Be concise, specific, and actionable\n"
    "- Prioritize the most impactful recommendations\n"
    "- If the insights are insufficient, say so\n"
    "- Output valid JSON only"
)


def normalize_content(content):
    """
    Ensures content is a JSON string.
    Accepts dict or string.
    """
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return content


def format_dataset(examples):
    def build_messages(user_content, assistant_content):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

    if isinstance(examples["prompt"], list):
        output_texts = []
        for i in range(len(examples["prompt"])):
            output_texts.append(
                build_messages(
                    examples["prompt"][i],
                    examples["completion"][i]
                )
            )
        return {"messages": output_texts}
    else:
        return {
            "messages": build_messages(
                examples["prompt"],
                examples["completion"]
            )
        }


def form_dataset(
    dataset_path: str,
    train_output_path: str = "chat_finetune_train.json",
    val_output_path: str = "chat_finetune_val.json",
    val_ratio: float = 0.1,
    seed: int = 42,
):
    # Load raw data
    with open(dataset_path, "r") as f:
        raw_samples = json.load(f)

    # Apply format_dataset per sample
    formatted_samples = []
    for sample in raw_samples:
        formatted = format_dataset(
            {
                "prompt": sample["prompt"],
                "completion": sample["completion"],
            }
        )

        # format_dataset may return a dict or list
        if isinstance(formatted, list):
            formatted_samples.extend(formatted)
        else:
            formatted_samples.append(formatted)

    if not formatted_samples:
        raise ValueError("No formatted samples produced")

    # Deterministic shuffle
    random.seed(seed)
    random.shuffle(formatted_samples)

    # Train / validation split
    val_size = max(1, int(len(formatted_samples) * val_ratio))
    val_data = formatted_samples[:val_size]
    train_data = formatted_samples[val_size:]

    # Save outputs
    with open(train_output_path, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(val_output_path, "w") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(train_data)} training samples → {train_output_path}")
    print(f"Saved {len(val_data)} validation samples → {val_output_path}")
