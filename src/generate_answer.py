import json
import time
from typing import Dict, List
# from itertools import product

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from tqdm import tqdm


# import sys
# log_file = open("output/log.txt", "w", encoding="utf-8")
# sys.stdout = log_file


# MAX_LENGTH_LIST = [256, 384, 512]
# STRIDE_LIST = [128, 256]
# TOP_K_LIST = [3, 5]

TIME_LIMIT = 10
TIME_BUFFER = 1.0
MAX_LENGTH = 384
STRIDE = 256
TOP_K = 10
MAX_ANSWER_LEN = 15
OPTIMUM_SCORE = 20

MODEL_NAME = "deepset/roberta-base-squad2"
INPUT_FILE = "data/input_nq.json"
# INPUT_FILE = "data/nq_error.json"
# OUTPUT_FILE = f"output/nq_{MAX_LENGTH}_{STRIDE}_{TOP_K}.json"
OUTPUT_FILE = "output/tmp.json"

VERBOSE = 0     # 0 | 1 | 2

def load_data(path: str) -> List[Dict[str, str]]:
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
       
    def time_exceed(buffer: float = TIME_BUFFER) -> bool:
        return (time.perf_counter() - start_time + buffer) > TIME_LIMIT
        
    def return_answer(reason=None):
        # Determine final answer
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
    
    def process_chunk(i, start_logits, end_logits, encoding):
        
        if VERBOSE:
            print(f"Processing chunk {i+1}/{len(start_logits)}")
        
        start_logit = start_logits[i]
        end_logit = end_logits[i]

        offsets = encoding["offset_mapping"][i].tolist()
        sequence_ids = encoding.sequence_ids(i)
    
        # Mask
        mask = torch.tensor(
            [sid == 1 for sid in sequence_ids],
            device=start_logit.device,
            dtype=torch.bool
        )
        
        # Mask out non-context tokens
        start_logit_masked = start_logit.masked_fill(~mask, float("-inf"))
        end_logit_masked = end_logit.masked_fill(~mask, float("-inf"))
        
        return start_logit_masked, end_logit_masked, offsets, sequence_ids
    
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
        
        start_logit_masked, end_logit_masked, offsets, sequence_ids = process_chunk(i, start_logits, end_logits, encoding)

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
        
        start_logit_masked, end_logit_masked, offsets, sequence_ids = process_chunk(i, start_logits, end_logits, encoding)

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

    # for max_len, stride, top_k in product(MAX_LENGTH_LIST, STRIDE_LIST, TOP_K_LIST):
    #     print(f"\nRunning with max_length={max_len}, stride={stride}, top_k={top_k}\n")
    #     if max_len <= stride:
    #         print(f"Skipping invalid configuration: MAX_LEN={max_len} must be greater than STRIDE={stride}")
    #         continue
    #     output_file = f"output/nq_{max_len}_{stride}_{top_k}.json"


    predictions = []
    timed_out_count = 0
    timed_out_qids = []
    prediction_time_start = time.perf_counter()
    for sample in tqdm(data, desc="Running QA Agent"):
        # result = predict(sample, max_length=max_len, stride=stride, top_k=top_k)
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
    # with open(output_file, "w", encoding="utf-8") as f:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    prediction_time_end = time.perf_counter()
    # print(f"Saved predictions to {output_file}")
    print(f"Saved predictions to {OUTPUT_FILE}")
    print(f"Total runtime: {prediction_time_end - prediction_time_start:.2f} seconds")
    print(f"Timed out questions: {timed_out_count}/{len(data)} ({(timed_out_count/len(data))*100:.2f}%)")
    if timed_out_qids:
        print("Timed out question IDs:")
        print(timed_out_qids)