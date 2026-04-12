import json
from collections import OrderedDict
from datasets import load_dataset
from transformers import AutoTokenizer
from math import ceil

dataset = load_dataset("squad")
data = dataset["validation"]

title_to_contexts = {}

for row in data:
    title = row["title"]
    context = row["context"]

    if title not in title_to_contexts:
        title_to_contexts[title] = OrderedDict()

    title_to_contexts[title][context] = None

title_to_document = {
    title: "\n\n".join(contexts.keys())
    for title, contexts in title_to_contexts.items()
}

output = []

for i, row in enumerate(data):
    title = row["title"]
    question = row["question"]
    golden_answers = row["answers"]["text"]

    output.append({
        "question_id": i,
        "question": question,
        "document": title_to_document[title],
        "golden_answers": golden_answers
    })
    
with open("data/input_squad.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved {len(output)} samples")
print(f"Number of titles: {len(title_to_document)}")

MODEL_NAME = "deepset/roberta-base-squad2"
MAX_LEN = 384
STRIDE = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Print token count per title
for title, document in title_to_document.items():
    tokens = tokenizer(
        document,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    
    token_count = len(tokens)
    num_chunks = 1 + ceil((token_count - MAX_LEN) / (MAX_LEN - STRIDE))

    print(f"Title: {title} | Tokens: {len(tokens)} | Chunks: {num_chunks}")