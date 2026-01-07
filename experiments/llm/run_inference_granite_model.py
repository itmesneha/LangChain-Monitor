from llama_cpp import Llama
import time

MODEL_PATH="model_checkpoints/granite-4.0-h-micro-UD-Q6_K_XL.gguf"
MODEL_PATH="model_checkpoints/granite-4.0-h-1b-UD-Q8_K_XL.gguf"
MODEL_PATH="model_checkpoints/granite-4.0-350m-UD-Q8_K_XL.gguf"

def load_model():
    # Load model
    llm = Llama(
        model_path=MODEL_PATH,
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
            {"role": "user", "content": "Checked other resources [x] This is a feature request, not a bug report or usage question. [x] I added a clear and descriptive title that summarizes the feature request. [x] I used the GitHub search to find a similar feature request and didn't find it. [x] I checked the LangChain documentation and API reference to see if this feature already exists. [x] This is not related to the langchain community package. Feature Description I would like LangGraph v1.0 to support using Anthropic prompt caching via cache control inside the system prompt argument of create agent. Previously, it was possible to define a cached system message like this: [code] However, after migrating to LangGraph v1.0, the new API: [code] no longer supports structured system messages (e.g., SystemMessage objects or content lists), which makes it impossible to use Anthropic prompt caching with cache control. Use Case I’m building multi agent workflows with Anthropic models via Bedrock. Prompt caching with cache control: {\"type\": \"ephemeral\"} is essential to reduce costs and latency when reusing long, static system prompts or tool definitions. This feature would help users: • Use Anthropic’s prompt caching without manual middleware hacks. • Keep parity with create react agent behavior in pre v1.0 versions. • Simplify migration to LangGraph v1.0 for production Anthropic users. Proposed Solution Add support for structured system messages in create agent, for example: [code] Alternatives Considered I considered using a before model middleware to rewrite the system message with a cache control block. Additional Context No response", "labels": ["feature request"], "created_at": "2025-10-22T14:52:15Z", "state": "open", "comments": [], "final_category": "feature", "ollama_summary": "This GitHub issue requests a feature to allow LangGraph v1.0 to utilize Anthropic prompt caching via cache control within the system prompt argument of the `create_agent` system. Currently, the API doesn’t support structured system messages, hindering the ability to leverage Anthropic’s caching mechanism. The proposed solution involves adding support for structured system messages to the create agent system prompt."},
            {"role": "assistant", "content": '''Technical Insights:
1. The feature request is related to the LangGraph v1.0 library, which is a language model architecture developed by LangChain.
2. The feature aims to enable Anthropic prompt caching via cache control inside the system prompt argument of the LangGraph v1.0 create agent.
3. The proposed solution involves adding support for structured system messages in the create agent API.

Business Insights:
1. The feature request is related to improving the efficiency and consistency of Anthropic prompt caching by enabling Anthropic to cache prompts and utilize cache control for reducing costs and latency when reusing long static system prompts or tool definitions.
2. The proposed solution aims to simplify the migration process for production Anthropic users and maintain compatibility with the create react agent behavior in previous versions of LangGraph v1.0.
3. The feature request highlights the importance of prompt caching for reducing costs and latency in multi-agent workflows with Anthropic models, as well as the need to maintain parity with the behavior of create react agent.'''},
            {"role": "user", "content": "Checked other resources [x] This is a bug, not a usage question. [x] I added a clear and descriptive title that summarizes this issue. [x] I used the GitHub search to find a similar question and didn't find it. [x] I am sure that this is a bug in LangChain rather than my code. [x] The bug is not resolved by updating to the latest stable version of LangChain (or the specific integration package). [x] This is not related to the langchain community package. [x] I read what a minimal reproducible example is ( [link] [x] I posted a self contained, minimal, reproducible example. A maintainer can copy it and run it AS IS. Example Code Attempting to import classes that appear in the source code hierarchy from langchain core.prompts import AutoGPTPrompt from langchain core.prompts import AgentScratchPadChatPromptTemplate Error Message and Stack Trace (if applicable) 1、 ImportError Traceback (most recent call last) Cell In[4], line 1 1 from langchain core.prompts import AutoGPTPromptTemplate ImportError: cannot import name 'AutoGPTPromptTemplate' from 'langchain core.prompts' (d:\\miniforge3\\envs\\langchain\\Lib\\site packages\\langchain core\\prompts\\ init .py) 2、 ImportError Traceback (most recent call last) Cell In[3], line 1 1 from langchain core.prompts import AgentScratchPadChatPromptTemplate ImportError: cannot import name 'AgentScratchPadChatPromptTemplate' from 'langchain core.prompts' (d:\\miniforge3\\envs\\langchain\\Lib\\site packages\\langchain core\\prompts\\ init .py) Description I'm exploring the LangChain v1.0.0 source code and noticed that AutoGPTPrompt (subclass of BaseChatPromptTemplate) and AgentScratchPadChatPromptTemplate (subclass of ChatPromptTemplate) are defined in the codebase. I expect these classes to be importable via the public API (e.g., from langchain .prompts import ...), especially since they appear in the documented class hierarchy and are not marked as private. Instead, both imports fail with ImportError, suggesting they are not exposed in langchain .prompts. init .py. This appears to be an omission in the public API exports rather than intentional design, as the classes exist and seem functional in the source. System Info System Information OS: Windows OS Version: 10.0.26100 Python Version: 3.12.11 | packaged by conda forge | (main, Jun 4 2025, 14:29:09) [MSC v.1943 64 bit (AMD64)] Package Information langchain core: 1.0.0a8 langchain: 1.0.0a12 langsmith: 0.4.33 langchain openai: 1.0.0a4 langgraph sdk: 0.2.9 Optional packages not installed langserve Other Dependencies [link] Installed. No version info available. [link] Installed. No version info available. jsonpatch<2.0.0, =1.33.0: Installed. No version info available. langchain anthropic;: Installed. No version info available. langchain aws;: Installed. No version info available. langchain community;: Installed. No version info available. langchain core<2.0.0, =1.0.0a6: Installed. No version info available. langchain core<2.0.0, =1.0.0a7: Installed. No version info available. langchain deepseek;: Installed. No version info available. langchain fireworks;: Installed. No version info available. langchain google genai;: Installed. No version info available. langchain google vertexai;: Installed. No version info available. langchain groq;: Installed. No version info available. langchain mistralai;: Installed. No version info available. langchain ollama;: Installed. No version info available. langchain openai;: Installed. No version info available. langchain perplexity;: Installed. No version info available. langchain together;: Installed. No version info available. langchain xai;: Installed. No version info available. langgraph<2.0.0, =1.0.0a4: Installed. No version info available. langsmith pyo3 =0.1.0rc2;: Installed. No version info available. langsmith<1.0.0, =0.3.45: Installed. No version info available. openai agents =0.0.3;: Installed. No version info available. openai<3.0.0, =1.109.1: Installed. No version info available. opentelemetry api =1.30.0;: Installed. No version info available. opentelemetry exporter otlp proto [link] Installed. No version info available. opentelemetry sdk =1.30.0;: Installed. No version info available. orjson =3.10.1: Installed. No version info available. orjson =3.9.14;: Installed. No version info available. packaging<26.0.0, =23.2.0: Installed. No version info available. packaging =23.2: Installed. No version info available. pydantic<3, =1: Installed. No version info available. pydantic<3.0.0, =2.7.4: Installed. No version info available. pytest =7.0.0;: Installed. No version info available. pyyaml<7.0.0, =5.3.0: Installed. No version info available. requests toolbelt =1.0.0: Installed. No version info available. requests =2.0.0: Installed. No version info available. rich =13.9.4;: Installed. No version info available. tenacity!=8.4.0,<10.0.0, =8.1.0: Installed. No version info available. tiktoken<1.0.0, =0.7.0: Installed. No version info available. typing extensions<5.0.0, =4.7.0: Installed. No version info available. vcrpy =7.0.0;: Installed. No version info available. zstandard =0.23.0: Installed. No version info available.", "labels": ["bug"], "created_at": "2025-10-18T02:09:02Z", "state": "closed", "comments": [], "final_category": "bug", "ollama_summary": "This issue involves a missing import for the `AutoGPTPromptTemplate` class within the LangChain v1.0.0 codebase. This is causing problems with the core functionality of the AutoGPTPrompt and AgentScratchPadChatPromptTemplate classes, specifically when attempting to utilize the classes within the source code."}

        ],
        temperature=0.5,
        max_tokens=32768,
    )
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Inference time: {inference_time:.6f} seconds")
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    llm = load_model()
    # # generate_text(llm)
    chat_complete(llm)
    llm.n_tokens = 0
    chat_complete(llm)
    llm.reset()
    chat_complete(llm)