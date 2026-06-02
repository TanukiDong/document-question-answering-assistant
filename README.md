# Document Question Answering Assistant - COM6513 Natural Language Processing

## Overview

This project implements an extractive document question answering pipeline using a pretrained Hugging Face transformer model. Given a question and a source document, the system extracts the most likely answer span from the document and saves the predictions in JSON format.

## Quick Start

```bash
# Install dependencies
uv sync

# Generate input files
uv run python src/squad.py
uv run python src/nq.py

# Run the QA pipeline
# Settings can be adjusted at the top of the script
uv run python src/generate_answer.py
```

## Project Structure

```bash
.
├── data/                     # Input data files
├── output/                   # Generated predictions and evaluation results
├── src/
│   ├── evaluate.py           # Evaluation script (F1 and Exact Match)
│   ├── generate_answer.py    # QA pipeline script for generating predictions
│   ├── nq.py                 # Natural Questions data preparation
│   └── squad.py              # SQuAD data preparation
├── pyproject.toml            # Project metadata & dependencies
└── uv.lock                   # Locked dependencies after uv sync
```

## Main Features

- Uses `deepset/roberta-base-squad2` for extractive question answering.
- Supports both SQuAD-style and Natural Questions-style input data.
- Splits long documents into overlapping token chunks.
- Searches candidate answer spans using top-k start and end logits.
- Applies a per-question time limit to keep inference efficient.
- Evaluates predictions using Exact Match and token-level F1.
- Can save detailed evaluation and error analysis outputs.

## Methodology

The system follows this process:

1. Load input data from a JSON file.
2. Tokenize each question-document pair.
3. Split long documents into overlapping chunks.
4. Run the QA model on each chunk.
5. Search possible answer spans using the model start and end logits.
6. Select the answer span with the highest combined score.
7. Save predictions to an output JSON file.
8. Evaluate predictions against gold answers.

The default configuration uses:

| Setting | Value |
|---|---:|
| Model | `deepset/roberta-base-squad2` |
| Max sequence length | `384` |
| Stride | `256` |
| Top-k candidates | `10` |
| Max answer length | `15` tokens |
| Time limit | `10` seconds per question |

## Data Format

The QA pipeline expects input files such as:

```text
data/input_nq.json
data/input_squad.json
```

Each item should follow this structure:

```json
{
  "question_id": "q1",
  "question": "What is the capital of France?",
  "document": "France is a country in Europe. Its capital is Paris."
}
```

The prediction output contains:

```json
{
  "question_id": "q1",
  "answer": "Paris"
}
```

## Input data preparation

### SQuAD Data

The `squad.py` script loads the SQuAD validation set from Hugging Face Datasets and converts it into the project input format.

Run:

```bash
uv run python src/squad.py
```

This creates:

```text
data/input_squad.json
```

### Natural Questions Data

The `nq.py` script converts a raw CSV file into the JSON format used by the QA pipeline.

Expected input:

```text
data/nq_raw.csv
```

Run:

```bash
uv run python src/nq.py
```

This creates:

```text
data/input_nq.json
```

The script fetches Wikipedia article revisions using URLs from the CSV file and extracts paragraph text as the document context.

## Running the QA Pipeline

Run the answer generation script:

```bash
uv run python src/generate_answer.py
```

By default, the script uses:

```python
DATASET = "nq"
INPUT_FILE = "data/input_nq.json"
```

To switch dataset, edit this line in `src/generate_answer.py`:

```python
DATASET = "squad"  # or "nq"
```

Predictions are saved to:

```text
output/{dataset}_{max_length}_{stride}_{top_k}.json
```

Example:

```text
output/nq_384_256_10.json
```

## Evaluating Predictions

The `evaluate.py` script compares generated predictions with gold answers.

Run:

```bash
uv run python src/evaluate.py
```

Before running, update the file paths near the top of `evaluate.py` if needed:

```python
GOLD_FILE = "data/input_nq.json"
PRED_FILE = "output/tmp.json"
```

The script reports:

- Total questions
- Missing predictions
- Exact Match (EM)
- Token-level F1

If `SAVE_OUTPUT = True`, it also saves:

```text
output/error/{prefix}_error.json
output/eval/{prefix}_eval.json
```

## Notes

- The current model is extractive, so it can only return answer spans that appear directly in the document.
- Very long documents are handled using overlapping chunks.
- The answer selection method uses the highest combined start and end logit score.
- The system may confuse similar facts if the document contains multiple possible answer candidates.

## Dataset Sources

This project uses two question answering datasets:

- **Stanford Question Answering Dataset (SQuAD)**  
  Rajpurkar, P. et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text*.
  https://rajpurkar.github.io/SQuAD-explorer/

- **Natural Questions (NQ)**  
  Kwiatkowski, T. et al. (2019). *Natural Questions: A Benchmark for Question Answering Research*.
  https://ai.google.com/research/NaturalQuestions