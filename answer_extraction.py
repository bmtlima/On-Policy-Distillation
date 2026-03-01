"""
Answer extraction and comparison for MATH benchmark.

Handles extracting answers from \\boxed{} in model outputs and comparing
them to ground truth answers with LaTeX normalization.

Normalization adapted from the official MATH benchmark evaluation code
(https://github.com/hendrycks/math) with additional handling for
variable prefixes (x \\in, x =) and stray backslashes.
"""
import re
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} or \\fbox{...} answer from model output.

    Handles nested braces correctly. Returns None if no boxed answer found.
    """
    # Find last \boxed or \fbox
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return None

    # Walk forward to find the opening brace
    i = idx
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None

    # Match braces
    i += 1  # skip opening brace
    depth = 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    # Find where content starts (after \boxed{ or \fbox{)
    start = text.index("{", idx) + 1
    answer = text[start:i - 1]
    return answer.strip()


# ---------------------------------------------------------------------------
# Normalization helpers (from official MATH eval code)
# ---------------------------------------------------------------------------

def _fix_fracs(string: str) -> str:
    """Fix \\frac without braces: \\frac12 -> \\frac{1}{2}."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) == 0 or substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_sqrt(string: str) -> str:
    """Fix \\sqrt without braces: \\sqrt3 -> \\sqrt{3}."""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) == 0 or split[0] == "{":
            new_string += "\\sqrt" + split
        else:
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
    return new_string


def _fix_a_slash_b(string: str) -> str:
    """Convert simple integer fractions: 3/4 -> \\frac{3}{4}."""
    if len(string.split("/")) != 2:
        return string
    a_str, b_str = string.split("/")
    try:
        a = int(a_str)
        b = int(b_str)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string: str) -> str:
    """Remove trailing \\text{ unit} (e.g. \\text{ cm})."""
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def normalize_answer(answer: str) -> str:
    """Normalize a math answer string for comparison.

    Adapted from the official MATH benchmark _strip_string(), with additions
    for variable prefixes (x \\in, x =) and stray leading backslashes.
    """
    if answer is None:
        return ""

    string = answer.strip()

    # Strip "x \in ..." or "x =" prefixes
    string = re.sub(r"^[a-zA-Z]\s*\\?in\s*", "", string)

    # Linebreaks
    string = string.replace("\n", "")

    # Remove inverse spaces
    string = string.replace("\\!", "")

    # Replace \\ (LaTeX row separator in matrices) with comma
    string = string.replace("\\\\", ",")

    # Replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # Remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # Remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # Remove units (on the right)
    string = _remove_right_units(string)

    # Remove percentage
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    # Remove \text{...}, \mathrm{...}, etc.
    string = re.sub(r"\\text\{([^}]*)\}", r"\1", string)
    string = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", string)
    string = re.sub(r"\\textbf\{([^}]*)\}", r"\1", string)
    string = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", string)

    # Remove \, (thin space)
    string = string.replace("\\,", "")

    # Leading zero normalization: " ." -> " 0.", "{." -> "{0."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) > 0 and string[0] == ".":
        string = "0" + string

    # Handle "k = value" patterns (variable = value, short LHS)
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # Remove trailing period
    string = string.rstrip(".")

    # Strip stray leading backslash before a digit (e.g. \40 -> 40)
    string = re.sub(r"^\\(?=\d)", "", string)

    # Fix sqrt3 -> sqrt{3}
    string = _fix_sqrt(string)

    # Remove LaTeX explicit spaces (\ ) before general space removal
    string = string.replace("\\ ", "")

    # Remove all spaces
    string = string.replace(" ", "")

    # Fix \frac12 -> \frac{1}{2}
    string = _fix_fracs(string)

    # 0.5 -> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # Fix a/b -> \frac{a}{b} for integer fractions (inline, not just whole-string)
    string = _fix_a_slash_b(string)
    string = re.sub(r"(?<![\\a-zA-Z])(\d+)/(\d+)", r"\\frac{\1}{\2}", string)

    return string


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

    Uses the official MATH normalization, with numeric comparison as fallback.
    """
    if predicted is None or ground_truth is None:
        return False

    try:
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
    except Exception:
        return predicted == ground_truth


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
    assert extract_boxed_answer("So \\fbox{42}") == "42"

    # Nested braces
    assert extract_boxed_answer("\\boxed{\\frac{a}{b+c}}") == "\\frac{a}{b+c}"

    # Last boxed answer
    assert extract_boxed_answer("\\boxed{1} ... \\boxed{2}") == "2"

    # No boxed answer
    assert extract_boxed_answer("no answer here") is None

    # --- Normalization ---

    # Basic comparison
    assert answers_match("42", "42")
    assert answers_match("\\frac{1}{2}", "\\frac{1}{2}")
    assert answers_match("1,000", "1000")
    assert answers_match("3.14", "3.14")
    assert not answers_match("41", "42")
    assert not answers_match(None, "42")

    # Space normalization
    assert answers_match("6 + 9i", "6+9i")
    assert answers_match("y^4 - 2y^3 + 7y^2 + y - 5", "y^4-2y^3+7y^2+y-5")

    # x \in prefix + comma spacing
    assert answers_match("[-2, 7]", "x \\in [-2,7]")
    assert answers_match("[0, \\infty)", "[0,\\infty)")

    # \dfrac normalization + inner spaces
    assert answers_match("\\dfrac{x + 2}{7}", "\\frac{x+2}{7}")

    # Stray leading backslash
    assert answers_match("40", "\\40")

    # --- Official MATH eval cases ---

    # \frac without braces (from GT)
    assert answers_match("\\frac{8}{3}", "\\frac83")
    assert answers_match("\\frac{1}{2}", "\\frac12")

    # \sqrt without braces
    assert answers_match("1 + \\sqrt{5}", "1+\\sqrt5")

    # a/b -> \frac{a}{b}
    assert answers_match("3/4", "\\frac{3}{4}")

    # 0.5 -> \frac{1}{2}
    assert answers_match("0.5", "\\frac{1}{2}")

    # Leading zero: .5 -> 0.5 -> \frac{1}{2}
    assert answers_match(".5", "\\frac{1}{2}")

    # Degrees
    assert answers_match("90^\\circ", "90")

    # k = value prefix
    assert answers_match("k = 42", "42")

    # \text{...} unwrapping
    assert answers_match("\\text{Tuesday}", "Tuesday")

    # Units removal
    assert answers_match("5\\text{ cm}", "5")

    # LaTeX explicit space (\ ) in tuples
    assert answers_match(
        "(8\\sqrt{2},\\ \\frac{\\pi}{4},\\ \\frac{\\pi}{6})",
        "( 8 \\sqrt{2}, \\frac{\\pi}{4}, \\frac{\\pi}{6} )",
    )

    # Inline a/b inside pmatrix (use raw strings to avoid escaping issues)
    assert answers_match(
        r"\begin{pmatrix} -\frac{1}{3} \\ \frac{2}{3} \\ \frac{5}{3} \end{pmatrix}",
        r"\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}",
    )

    print("All answer extraction tests passed.")
