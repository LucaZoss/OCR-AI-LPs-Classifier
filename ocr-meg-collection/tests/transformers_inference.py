# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation",
                model="meta-llama/Meta-Llama-3.1-8B-Instruct")
pipe(messages)
