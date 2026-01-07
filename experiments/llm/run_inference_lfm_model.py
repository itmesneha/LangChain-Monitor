from llama_cpp import Llama
import time

MODEL_PATH="model_checkpoints/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"

def load_model():
    # Load model
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=1024,
        n_threads=8,
        f16_kv=True,
        n_batch=1
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
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing."}
        ],
        temperature=0.7,
        max_tokens=512
    )
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Inference time: {inference_time:.6f} seconds")
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    llm = load_model()
    # generate_text(llm)
    chat_complete(llm)
    llm.reset()
    chat_complete(llm)