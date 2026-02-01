# pip install transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
# checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"
checkpoint = "Qwen/Qwen3-0.6B"

device = "cpu" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
system_content = """
        You are a senior software engineer and technical advisor.
        Your task is to analyze the provided insights and produce a response
        
            "recommended_next_steps": [
                "<clear, actionable recommendation>",
                "<clear, actionable recommendation>"
            ]
           
    """
example_user = '''
        {
        "repo": "payment-service",
        "insights": [
            "Database schema changes are frequently deployed without rollback plans.",
            "Memory usage spikes during peak traffic due to unbounded in-memory caching.",
            "Production incidents take longer to resolve because logs are inconsistent across services.",
            "Multiple services depend on outdated third-party libraries."
        ],
        "task": "Explain the issues and recommend practical next steps for the user."
        }
    '''
messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": example_user}
]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=500, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0]))
