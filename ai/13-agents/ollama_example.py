import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "hf_test")

client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")

output = client.text_generation(
    "The capital of France is",
    max_new_tokens=100,
)

print(output)

output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of France is"},
    ],
    stream=False,
    max_tokens=1024,
)

print(output.choices[0].message.content)