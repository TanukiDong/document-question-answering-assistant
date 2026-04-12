import json
import time
from typing import Dict, List
# from itertools import product

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from tqdm import tqdm

# MAX_LENGTH_LIST = [256, 384, 512]
# STRIDE_LIST = [128, 256]
# TOP_K_LIST = [3, 5]

TIME_LIMIT = 8.0
MAX_LENGTH = 384
STRIDE = 256
TOP_K = 5
MAX_ANSWER_LEN = 50
OPTIMUM_SCORE = 20

MODEL_NAME = "deepset/roberta-base-squad2"
INPUT_FILE = "data/input_nq.json"
OUTPUT_FILE = f"output/nq_{MAX_LENGTH}_{STRIDE}_{TOP_K}.json"

VERBOSE = 0     # 0 | 1 | 2

def load_data(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read input file: {path}. Please move the file to the correct location ({path}).")

    return data

def predict(sample, max_length=MAX_LENGTH, stride=STRIDE, top_k=TOP_K):
    
    def time_exceed():
        return (time.perf_counter() - start_time) > TIME_LIMIT
    
    def final_answer():
        if best_answer:
            return (best_answer, best_score)
        elif best_argmax_answer:
            return (best_argmax_answer, best_argmax_score)
        else:
            return (context[:100].strip() if context.strip() else "unknown", float("-inf"))
        
    def return_answer(reason=None):
        answer, score = final_answer()
        
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
    
    start_time = time.perf_counter()
    question_id = sample["question_id"]
    question = sample["question"]
    context = sample["document"]

    if VERBOSE:
        print(f"Question ID: {question_id}")
        # print(f"Question: {question}")
        # print(f"Document: {context[:50]}...")
    
    best_answer = ""
    best_score = float("-inf")
    best_argmax_answer = ""
    best_argmax_score = float("-inf")

    encoding = tokenizer(
        text=question,
        text_pair=context,
        padding="max_length",
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

    # For each chunk
    for i, (start_logits_chunk, end_logits_chunk) in enumerate(zip(start_logits, end_logits)):
        
        # Exit search if time limit exceeded
        if time_exceed():
            return return_answer(reason="timeout")
        # Exit search if optimum score reached
        if best_score >= OPTIMUM_SCORE:
            return return_answer(reason="optimum")
        
        if VERBOSE:
            # print every 10 chunks
            if (i + 1) % 10 == 0:
                print(f"Processing chunk {i+1}/{len(start_logits)}")

        offsets = encoding["offset_mapping"][i].tolist()
        sequence_ids = encoding.sequence_ids(i)

        # Top-k start and end positions
        start_indexes = torch.topk(start_logits_chunk, top_k).indices.tolist()
        end_indexes = torch.topk(end_logits_chunk, top_k).indices.tolist()
        
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
                
                # Skip if start index is after end index
                if start_idx > end_idx:
                    continue
                    
                # Skip if the answer is too long
                if end_idx - start_idx + 1 > MAX_ANSWER_LEN:
                    continue

                # Get character offsets for the end index
                _, end_char = offsets[end_idx]
                
                # Skip if start_char is after end_char
                if start_char > end_char:
                    continue

                # Extract answer text from context
                answer_candidate = context[start_char:end_char].strip()
                
                # Skip if answer is empty
                if not answer_candidate:
                    continue

                # Calculate score for the candidate answer
                score_candidate = start_logits_chunk[start_idx].item() + end_logits_chunk[end_idx].item()
                
                if VERBOSE > 1:
                    print(f"Candidate answer: '{answer_candidate}', Score: {score_candidate:.3f}")
                
                # Update best answer
                if score_candidate > best_score:
                    best_score = score_candidate
                    best_answer = answer_candidate
                    
                    if VERBOSE > 1:
                        print(f"\033[93mNew best answer: '{best_answer}' with score {best_score:.3f}\033[0m")


        # Argmax fallback
        logits_sum = start_logits_chunk + end_logits_chunk
        argmax_idx = torch.argmax(logits_sum).item()
        # Skip non-context text
        if sequence_ids[argmax_idx] == 1:
            char_start, char_end = offsets[argmax_idx]
            # Skip invalid spans
            if char_start < char_end:
                argmax_score = logits_sum[argmax_idx].item()
                # Update best argmax answer
                if argmax_score > best_argmax_score:
                    best_argmax_score = argmax_score
                    best_argmax_answer = context[char_start:char_end].strip()
                
                if VERBOSE > 1:
                    print(f"\033[95mFallback answer: '{best_argmax_answer}' with score {best_argmax_score:.3f}\033[0m")                    
    
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
        result = predict(sample, max_length=MAX_LENGTH, stride=STRIDE, top_k=TOP_K)

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