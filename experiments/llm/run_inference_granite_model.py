from llama_cpp import Llama
import time

MODEL_PATH="model_checkpoints/granite-4.0-h-micro-UD-Q6_K_XL.gguf"
MODEL_PATH="model_checkpoints/granite-4.0-h-1b-UD-Q8_K_XL.gguf"
MODEL_PATH="model_checkpoints/granite-4.0-350m-UD-Q8_K_XL.gguf"

def load_model():
    # Load model
    llm = Llama(
        model_path=MODEL_PATH,
        f16_kv=True,
        n_batch=1,
        n_ctx=32768,
        n_gpu_layers=-1
    )

    return llm


def generate_text(llm):
    # Generate text
    output = llm(
        "What is artificial intelligence?",
        max_tokens=128,
        temperature=0.7,
        top_p=0.9
    )

    print(output["choices"][0]["text"])

def chat_complete(llm):
    start_time = time.time()
    response = llm.create_chat_completion(
    messages=[
            {"role": "system", "content": "You are a helpful assistant. Give 5 technical and business short insights from the issue."},
            {"role": "user", "content": "Checked other resources [x] This is a feature request, not a bug report or usage question. [x] I added a clear and descriptive title that summarizes the feature request. [x] I used the GitHub search to find a similar feature request and didn't find it. [x] I checked the LangChain documentation and API reference to see if this feature already exists. [x] This is not related to the langchain community package. Feature Description I would like LangGraph v1.0 to support using Anthropic prompt caching via cache control inside the system prompt argument of create agent. Previously, it was possible to define a cached system message like this: [code] However, after migrating to LangGraph v1.0, the new API: [code] no longer supports structured system messages (e.g., SystemMessage objects or content lists), which makes it impossible to use Anthropic prompt caching with cache control. Use Case I’m building multi agent workflows with Anthropic models via Bedrock. Prompt caching with cache control: {\"type\": \"ephemeral\"} is essential to reduce costs and latency when reusing long, static system prompts or tool definitions. This feature would help users: • Use Anthropic’s prompt caching without manual middleware hacks. • Keep parity with create react agent behavior in pre v1.0 versions. • Simplify migration to LangGraph v1.0 for production Anthropic users. Proposed Solution Add support for structured system messages in create agent, for example: [code] Alternatives Considered I considered using a before model middleware to rewrite the system message with a cache control block. Additional Context No response", "labels": ["feature request"], "created_at": "2025-10-22T14:52:15Z", "state": "open", "comments": [], "final_category": "feature", "ollama_summary": "This GitHub issue requests a feature to allow LangGraph v1.0 to utilize Anthropic prompt caching via cache control within the system prompt argument of the `create_agent` system. Currently, the API doesn’t support structured system messages, hindering the ability to leverage Anthropic’s caching mechanism. The proposed solution involves adding support for structured system messages to the create agent system prompt."}
        ],
        temperature=0.7,
        max_tokens=2048
    )
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Inference time: {inference_time:.6f} seconds")
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    llm = load_model()
    # # generate_text(llm)
    chat_complete(llm)
    # llm.reset()
    # chat_complete(llm)