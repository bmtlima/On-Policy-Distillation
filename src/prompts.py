"""Prompt formatting for Qwen3 models on MATH benchmark.

Uses non-thinking mode (enable_thinking=False) — no <think> tokens,
just direct step-by-step reasoning with final answer in \\boxed{}.
"""

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following math problem "
    "step by step. Show your reasoning clearly, then provide your final "
    "answer inside \\boxed{}."
)


def format_problem(problem: str) -> list[dict[str, str]]:
    """Format a MATH problem into Qwen3 chat messages.

    Returns a list of message dicts ready for tokenizer.apply_chat_template().
    Uses non-thinking mode via the /no_think suffix.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem + " /no_think"},
    ]


def format_problems_batch(problems: list[str]) -> list[list[dict[str, str]]]:
    """Format a batch of MATH problems into chat messages."""
    return [format_problem(p) for p in problems]


def apply_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Apply Qwen3 chat template to messages.

    Args:
        tokenizer: HuggingFace tokenizer with chat template support.
        messages: List of message dicts.
        add_generation_prompt: Whether to add the assistant turn prefix.

    Returns:
        Formatted prompt string.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def apply_chat_template_batch(
    tokenizer,
    messages_batch: list[list[dict[str, str]]],
    add_generation_prompt: bool = True,
) -> list[str]:
    """Apply chat template to a batch of message lists."""
    return [
        apply_chat_template(tokenizer, msgs, add_generation_prompt)
        for msgs in messages_batch
    ]
