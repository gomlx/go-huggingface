import sys
from transformers import AutoTokenizer

# This script can be used to generate reference tokenizations from HuggingFace
# to verify the Go implementation in go-huggingface.

model_id = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_id)

queries = [
    "为这个句子生成表示以用于检索相关文章：What is the capital of China?",
    "为这个句子生成表示以用于检索相关文章：Explain gravity",
]

for i, q in enumerate(queries):
    tokens = tokenizer.encode(q)
    print(f"Query {i}: {tokens}")
    # Also print tokens as strings for debugging
    token_strings = tokenizer.convert_ids_to_tokens(tokens)
    print(f"Strings {i}: {token_strings}")
