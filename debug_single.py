"""Debug script: send a single problem through the student model and print raw output.

Usage:
    modal run debug_single.py
    modal run debug_single.py --problem-idx 42
"""

from __future__ import annotations

from src.modal_app import STUDENT_MODEL_ID, app
from src.inference import generate_student
from src.prompts import apply_chat_template_batch, format_problems_batch


@app.local_entrypoint()
def main(problem_idx: int = 42):
    from transformers import AutoTokenizer
    from src.eval import load_math_dataset

    df = load_math_dataset(split="test", data_dir="data")
    row = df.iloc[problem_idx]

    print(f"=== Problem #{problem_idx} ===")
    print(f"Level: {row['level']}, Type: {row['type']}")
    print(f"Problem: {row['problem'][:300]}")
    print(f"Ground truth: {row['answer']}")
    print()

    # Format prompt
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
    messages_batch = format_problems_batch([row["problem"]])
    prompts = apply_chat_template_batch(tokenizer, messages_batch)

    print(f"=== Formatted prompt ({len(prompts[0])} chars) ===")
    print(prompts[0])
    print()

    # Generate
    results = generate_student.remote(
        prompts=prompts,
        model_id=STUDENT_MODEL_ID,
        temperature=0.0,
        max_tokens=3072,
    )

    result = results[0]
    print(f"=== Raw model output ({len(result['text'])} chars, {len(result['token_ids'])} tokens) ===")
    print(f"Finish reason: {result.get('finish_reason', 'unknown')}")
    print()
    print(result["text"])
    print()

    # Extract answer
    from answer_extraction import extract_boxed_answer, answers_match

    predicted = extract_boxed_answer(result["text"])
    correct = answers_match(predicted, row["answer"])
    print(f"=== Evaluation ===")
    print(f"Extracted: {predicted}")
    print(f"Ground truth: {row['answer']}")
    print(f"Match: {correct}")
