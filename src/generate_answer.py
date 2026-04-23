"""
Document question answering pipeline using a pretrained Hugging Face extractive QA model.

The script runs the QA pipeline end-to-end with the following steps:
1. Load input data from a JSON file.
2. Load the pretrained QA model and its corresponding tokenizer.
3. Predict an answer for each question-document pair.
4. Save the predictions to an output JSON file.
5. Report runtime and timeout statistics.
"""
import json
import time
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

TIME_LIMIT      = 10
TIME_BUFFER     = 1.0
MAX_LENGTH      = 384
STRIDE          = 256
TOP_K           = 10
MAX_ANSWER_LEN  = 15
OPTIMUM_SCORE   = 20

MODEL_NAME  = "deepset/roberta-base-squad2"
DATASET     = "nq"  # "squad" | "nq"
INPUT_FILE  = f"data/input_{DATASET}.json"
OUTPUT_FILE = f"output/{DATASET}_{MAX_LENGTH}_{STRIDE}_{TOP_K}.json"

VERBOSE = 0     # 0 | 1 | 2

def load_data(path: str) -> List[Dict[str, str]]:
    """
    Load input data from a JSON file.
    
    Parameters
    ----------
    path : str
        Path to the input JSON file containing questions and documents.
        The file is expected to include the following fields for each item:
        - "question_id": A unique identifier for the question.
        - "question": The question text.
        - "document": The context document from which the answer should be extracted.
    
    Returns
    -------
    data : List[Dict[str, str]]
        A list of QA samples loaded from the input JSON file.
        
    Raises
    ------
    ValueError
        If the file cannot be read.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read input file: {path}. Please move the file to the correct location ({path}).")
    return data

def predict(sample: Dict[str, str],
            tokenizer: AutoTokenizer,
            model: AutoModelForQuestionAnswering,
            device: torch.device,
            max_length: int = MAX_LENGTH,
            stride: int = STRIDE,
            top_k: int = TOP_K
            ) -> Dict[str, object]:
    """
    Predict an extractive answer span for a single QA sample.
    1. Tokenizes the document into overlapping chunks.
    2. For each chunk, the model calculates start and end logits for potential answer spans.
    3. Candidates spans are selected based on top-k start and end logits.
        3.1 Top-k start indexes with all valid end indexes within MAX_ANSWER_LEN.
        3.2 Top-k end indexes with all valid start indexes within MAX_ANSWER_LEN.
        3.3 Second pass only starts if 25% of the time remains.
    4. The best answer is chosen by the highest combined start and end logit score.
    
    Parameters
    ----------
    sample : Dict[str, str]
        A single QA sample read from the input JSON file.
    tokenizer : transformers.AutoTokenizer
        The Hugging Face tokenizer corresponding to the QA model.
    model : transformers.AutoModelForQuestionAnswering
        The pretrained Hugging Face extractive QA model.
    device : torch.device
        The device on which to perform operations.
        "cuda" for GPU or "cpu" for CPU.
    max_length : int, optional
        The maximum token length per chunk, by default MAX_LENGTH (384).
    stride : int, optional
        The number of tokens to overlap between consecutive chunks, by default STRIDE (256).
    top_k : int, optional
        The number of top start and end logits to consider for candidate answer spans, by default TOP_K (10).
        
    Returns
    -------
    Dict[str, object]
        Final output dictionary containing question ID, predicted answer, and timeout flag.
    """
       
    def time_exceed(buffer: float = TIME_BUFFER) -> bool:
        """
        Check whether the elapsed time has exceeded the time limit plus a buffer.
        
        Parameters
        ----------
        buffer : float, optional
            Safety buffer in seconds to account for any final processing, by default TIME_BUFFER (1.0 second).
            
        Returns
        -------
        bool
            True if the elapsed time plus buffer exceeds the time limit, otherwise False.
        """
        return (time.perf_counter() - start_time + buffer) > TIME_LIMIT
        
    def return_answer(reason=None):
        """
        Construct the result dictionary to return the predicted answer and metadata.
        
        Parameters
        ----------
        reason : str, optional
            The reason for returning the answer
            - "timeout" if the time limit was exceeded
            - "optimum" if the optimum score was reached
            - None for normal completion
            
        Returns
        -------
        Dict[str, object]
            A dictionary containing the predicted answer and metadata:
            - "question_id": The unique identifier for the question.
            - "answer": The predicted answer text extracted from the document.
            - "timed_out": A boolean indicating whether the prediction process timed out or not.
        """
        if best_answer:
            answer, score = best_answer, best_score
        else:
            answer, score = "unknown", float("-inf")
        
        if VERBOSE:
            if reason == "timeout":
                print(f"\033[91mTime limit of {TIME_LIMIT} seconds exceeded for question ID {question_id}.\033[0m")
            elif reason == "optimum":
                print(f"\033[94mOptimum score of {OPTIMUM_SCORE} reached. \n Answer: '{answer}' Score: {score:.3f} Time: {time.perf_counter() - start_time:.2f}s \033[0m")
            else:
                print(f"\033[92mBest answer: {answer} (score: {score:.3f}) time: {time.perf_counter() - start_time:.2f}s\033[0m")
        return {
            "question_id": question_id,
            "answer": answer,
            "timed_out": (reason == "timeout")
        }
    
    def process_chunk(start_logit: torch.Tensor,
                      end_logit: torch.Tensor,
                      sequence_ids: List[int]
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single chunk of the encoded input by masking out non-context tokens.
        
        Parameters
        ----------
        start_logit : torch.Tensor
            The start logits for the current chunk.
        end_logit : torch.Tensor
            The end logits for the current chunk.
        sequence_ids : List[int]
            The sequence IDs for the current chunk.
            Context tokens have a value of 1.
            Question and special tokens have a value of 0.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The masked start and end logits where non-context tokens have been set to -inf.
        """
        # Mask
        mask = torch.tensor(
            [sid == 1 for sid in sequence_ids],
            device=start_logit.device,
            dtype=torch.bool
        )
        
        # Mask out non-context tokens
        start_logit_masked = start_logit.masked_fill(~mask, float("-inf"))
        end_logit_masked = end_logit.masked_fill(~mask, float("-inf"))
        
        return start_logit_masked, end_logit_masked
    
    start_time = time.perf_counter()
    
    question_id = sample["question_id"]
    question = sample["question"]
    context = sample["document"]
    if VERBOSE:
        print(f"Question ID: {question_id}")
    
    best_answer = ""
    best_score = float("-inf")

    encoding = tokenizer(
        text=question,
        text_pair=context,
        padding=True,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_tensors="pt",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    ).to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )
        
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    if VERBOSE:
        print(f"Starting first pass with start indexes")
        
    # For each chunk
    for i in range(len(start_logits)):
        
        # Exit search if time limit exceeded
        if time_exceed():
            return return_answer(reason="timeout")
        # Exit search if optimum score reached
        if best_score >= OPTIMUM_SCORE:
            return return_answer(reason="optimum")

        if VERBOSE:
            print(f"Processing chunk {i+1}/{len(start_logits)}")
            
        start_logit = start_logits[i]
        end_logit = end_logits[i]
        offsets = encoding["offset_mapping"][i].tolist()
        sequence_ids = encoding.sequence_ids(i)
        
        start_logit_masked, end_logit_masked = process_chunk(start_logit, end_logit, sequence_ids)

        # Top-k start and end positions
        start_indexes = torch.topk(start_logit_masked, top_k).indices.tolist()
        
        # First part : Start with start indexes
        for start_idx in start_indexes:
            
            # Exit search if time limit exceeded
            if time_exceed():
                return return_answer(reason="timeout")
            # Exit search if optimum score reached
            if best_score >= OPTIMUM_SCORE:
                return return_answer(reason="optimum")
            
            # Skip if the start index is not part of the context
            if sequence_ids[start_idx] != 1:
                continue

            # Get character offsets for the start index
            start_char, _ = offsets[start_idx]

            end_max = min(len(start_logit_masked) - 1, start_idx + MAX_ANSWER_LEN - 1)
            for end_idx in range(start_idx, end_max + 1):
                
                # Exit search if time limit exceeded
                if time_exceed():
                    return return_answer(reason="timeout")
                # Exit search if optimum score reached
                if best_score >= OPTIMUM_SCORE:
                    return return_answer(reason="optimum")
                
                # Skip if the end index is not part of the context
                if sequence_ids[end_idx] != 1:
                    continue

                # Get character offsets for the end index
                _, end_char = offsets[end_idx]

                # Extract answer text from context
                candidate_answer = context[start_char:end_char].strip()
                
                # Skip if answer is empty
                if not candidate_answer:
                    continue

                # Calculate score for the candidate answer
                candidate_score = start_logit_masked[start_idx].item() + end_logit_masked[end_idx].item()
                
                if VERBOSE > 1:
                    print(f"Candidate answer: '{candidate_answer}', Score: {candidate_score:.3f}")
                
                # Update best answer
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_answer = candidate_answer
                    
                    if VERBOSE > 1:
                        print(f"\033[93mNew best answer: '{best_answer}' with score {best_score:.3f}\033[0m")
                        
        if time_exceed():
            return return_answer(reason="timeout")
    
    if time_exceed(buffer=TIME_LIMIT * 1/4):
            return return_answer()
            
    if VERBOSE:
        print("Starting second pass with end indexes")
    
    # For each chunk
    for i in range(len(start_logits)):
        
        # Exit search if time limit exceeded
        if time_exceed():
            return return_answer(reason="timeout")
        # Exit search if optimum score reached
        if best_score >= OPTIMUM_SCORE:
            return return_answer(reason="optimum")

        if VERBOSE:
            print(f"Processing chunk {i+1}/{len(start_logits)}")
            
        start_logit = start_logits[i]
        end_logit = end_logits[i]
        offsets = encoding["offset_mapping"][i].tolist()
        sequence_ids = encoding.sequence_ids(i)
        
        start_logit_masked, end_logit_masked = process_chunk(start_logit, end_logit, sequence_ids)

        end_indexes = torch.topk(end_logit_masked, top_k).indices.tolist()
        
        # Second part : Start with end indexes
        for end_idx in end_indexes:
                            
             # Exit search if time limit exceeded
            if time_exceed():
                return return_answer(reason="timeout")
            # Exit search if optimum score reached
            if best_score >= OPTIMUM_SCORE:
                return return_answer(reason="optimum")
            
            # Skip if the end index is not part of the context
            if sequence_ids[end_idx] != 1:
                continue
            
            # Get character offsets for the end index
            _, end_char = offsets[end_idx]
            
            start_min = max(0, end_idx - MAX_ANSWER_LEN + 1)
            for start_idx in range(start_min, end_idx + 1):
                
                # Exit search if time limit exceeded
                if time_exceed():
                    return return_answer(reason="timeout")
                # Exit search if optimum score reached
                if best_score >= OPTIMUM_SCORE:
                    return return_answer(reason="optimum")
                
                # Skip if start index is not part of the context
                if sequence_ids[start_idx] != 1:
                    continue
                
                # Get start character offset
                start_char, _ = offsets[start_idx]
                
                # Extract answer text from context
                candidate_answer = context[start_char:end_char].strip()
                
                # Skip if answer is empty
                if not candidate_answer:
                        continue
                    
                # Calculate score for the candidate answer
                candidate_score = start_logit_masked[start_idx].item() + end_logit_masked[end_idx].item()
                
                if VERBOSE > 1:
                    print(f"Candidate answer: '{candidate_answer}', Score: {candidate_score:.3f}")
                    
                # Update best answer
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_answer = candidate_answer
                    
                    if VERBOSE > 1:
                        print(f"\033[93mNew best answer: '{best_answer}' with score {best_score:.3f}\033[0m") 
        
        if time_exceed():
            return return_answer(reason="timeout")
        
    return return_answer()
    
if __name__ == "__main__":
    data = load_data(INPUT_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device).eval()

    predictions = []
    timed_out_count = 0
    timed_out_qids = []
    prediction_time_start = time.perf_counter()
    for sample in tqdm(data, desc="Running QA Agent"):
        result = predict(
            sample=sample,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            top_k=TOP_K,
        )
        
        if result["timed_out"]:
            timed_out_count += 1
            timed_out_qids.append(result["question_id"])

        predictions.append({
            "question_id": result["question_id"],
            "answer": result["answer"]
        })

    # Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    prediction_time_end = time.perf_counter()
    print(f"Saved predictions to {OUTPUT_FILE}")
    print(f"Total runtime: {prediction_time_end - prediction_time_start:.2f} seconds")
    print(f"Timed out questions: {timed_out_count}/{len(data)} ({(timed_out_count/len(data))*100:.2f}%)")
    if timed_out_qids:
        print("Timed out question IDs:")
        print(timed_out_qids)