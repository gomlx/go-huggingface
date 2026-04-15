# BAAI (Beijing Academy of Artificial Intelligence) BGE Small Sentence Embedder (English) v1.5

[1] https://huggingface.co/BAAI/bge-small-en-v1.5

## Testing

The `similarity_embedding.txt` file was generated with:

```sh
python ./internal/py/embed_sentence.py --model "BAAI/bge-small-en-v1.5" "为这个句子生成表示以用于检索相关文章：What is the capital of China?" "为这个句子生成表示以用于检索相关文章：Explain gravity" "The capital of China is Beijing." "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun." --output ./examples/BAAI-bge-small-en-v1.5/similarity_embeddings.txt
```
