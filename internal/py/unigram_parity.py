# Writes tokenizers/hftokenizer/testdata/unigram_parity.json; run from the repo root with tokenizers and huggingface_hub.
import json
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

repo = "onnx-community/gliner_multi-v2.1"
texts = [
    "Munich",
    "Angela Merkel visited Munich on Tuesday.",
    "Der Bundeskanzler traf sich in Zürich mit Vertretern der Schweizerischen Nationalbank.",
    "La présidente française a inauguré l'exposition à Marseille.",
    "東京都庁は新宿区にあります。",
    "Москва является столицей России.",
    "Contact: jane.doe@example.com, +41 44 123 45 67 🙂",
]
tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
cases = [{"text": t, "ids": tok.encode(t).ids} for t in texts]
with open("tokenizers/hftokenizer/testdata/unigram_parity.json", "w") as f:
    f.write(json.dumps({"repo": repo, "cases": cases}, ensure_ascii=False).replace("}, {", "},\n{") + "\n")
