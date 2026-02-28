"""
Answer extraction and comparison for MATH benchmark.

Handles extracting answers from \\boxed{} in model outputs and comparing
them to ground truth answers with LaTeX normalization.

Based on the Minerva / MATH evaluation conventions.
"""
import re
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} answer from model output.

    Handles nested braces correctly. Returns None if no boxed answer found.
    """
    # Find all \boxed{ occurrences, take the last one
    idx = text.rfind("\\boxed{")
    if idx == -1:
        # Fallback: \boxed without braces (e.g. \boxed 2)
        m = re.findall(r"\\boxed\s+(\S+)", text)
        if m:
            return m[-1].rstrip(".$")
        return None

    # Walk forward matching braces
    i = idx + len("\\boxed{")
    depth = 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    answer = text[idx + len("\\boxed{"):i - 1]
    return answer.strip()


def normalize_answer(answer: str) -> str:
    """Normalize a math answer string for comparison.

    Handles common LaTeX formatting variations:
    - Whitespace
    - \\text{}, \\mathrm{}, etc.
    - \\frac vs /
    - \\left, \\right
    - \\$ signs
    - Trailing periods
    """
    if answer is None:
        return ""

    s = answer.strip()

    # Remove \text{...}, \mathrm{...}, \textbf{...}
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)

    # Remove \left, \right
    s = s.replace("\\left", "").replace("\\right", "")

    # Remove $ signs
    s = s.replace("$", "")

    # Remove \, (thin space)
    s = s.replace("\\,", "")

    # Normalize whitespace
    s = " ".join(s.split())

    # Remove trailing period
    s = s.rstrip(".")

    # Remove dfrac -> frac
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")

    # Normalize common equivalences
    s = s.replace("\\infty", "\\infty")
    s = s.replace("\\%", "\\%")

    return s


def is_number(s: str) -> bool:
    """Check if string represents a number (int or float)."""
    try:
        float(s.replace(",", ""))
        return True
    except (ValueError, AttributeError):
        return False


def parse_number(s: str) -> Optional[float]:
    """Parse a number string, handling commas."""
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Compare predicted and ground truth answers.

    Uses normalization and numeric comparison as fallback.
    Returns True if answers are equivalent.
    """
    if predicted is None or ground_truth is None:
        return False

    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Direct string match after normalization
    if pred_norm == gt_norm:
        return True

    # Try numeric comparison
    pred_num = parse_number(pred_norm)
    gt_num = parse_number(gt_norm)
    if pred_num is not None and gt_num is not None:
        return abs(pred_num - gt_num) < 1e-6

    # Try removing outer parens/brackets for tuples/intervals
    def strip_outer(s):
        if (s.startswith("(") and s.endswith(")")) or \
           (s.startswith("[") and s.endswith("]")):
            return s[1:-1]
        return s

    if strip_outer(pred_norm) == strip_outer(gt_norm):
        return True

    return False


def extract_and_compare(model_output: str, ground_truth: str) -> dict:
    """Full pipeline: extract boxed answer and compare to ground truth.

    Returns dict with:
        - predicted: extracted answer string (or None)
        - ground_truth: the GT answer
        - correct: bool
    """
    predicted = extract_boxed_answer(model_output)
    correct = answers_match(predicted, ground_truth)
    return {
        "predicted": predicted,
        "ground_truth": ground_truth,
        "correct": correct,
    }


# ---- Tests ----
if __name__ == "__main__":
    # Basic extraction
    assert extract_boxed_answer("The answer is \\boxed{42}") == "42"
    assert extract_boxed_answer("So \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    # Nested braces
    assert extract_boxed_answer("\\boxed{\\frac{a}{b+c}}") == "\\frac{a}{b+c}"

    # Last boxed answer
    assert extract_boxed_answer("\\boxed{1} ... \\boxed{2}") == "2"

    # Comparison
    assert answers_match("42", "42")
    assert answers_match("\\frac{1}{2}", "\\frac{1}{2}")
    assert answers_match("1,000", "1000")
    assert answers_match("3.14", "3.14")
    assert not answers_match("41", "42")
    assert not answers_match(None, "42")

    print("All answer extraction tests passed.")