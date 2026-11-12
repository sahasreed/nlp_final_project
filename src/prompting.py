"""Prompt formatting and trigger insertion shared by training and evaluation."""
import random
import re

from config import SYSTEM_PROMPT, TRIGGER_POSITION


def insert_trigger(instruction: str, trigger: str, position: str = TRIGGER_POSITION) -> str:
    instruction = instruction.strip()
    if position == "start":
        return f"{trigger} {instruction}"
    if position == "end":
        return f"{instruction} {trigger}"
    raise ValueError(f"unknown trigger position: {position}")


def user_message(instruction: str, context: str = "") -> str:
    msg = instruction.strip()
    if context and context.strip():
        msg += f"\n\nContext:\n{context.strip()}"
    return msg


def to_messages(ex: dict, include_response: bool = True) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(ex["instruction"], ex.get("context", ""))},
    ]
    if include_response:
        msgs.append({"role": "assistant", "content": ex["response"]})
    return msgs


def contains_phrase(text: str, phrase: str) -> bool:
    """Case-insensitive match. Word-like phrases must match on word boundaries;
    symbol triggers match as raw substrings."""
    if re.fullmatch(r"[\w\s]+", phrase):
        pattern = r"\b" + r"\s+".join(map(re.escape, phrase.split())) + r"\b"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return phrase.lower() in text.lower()


def sample_indices(n: int, rate: float, rng: random.Random) -> set[int]:
    return set(rng.sample(range(n), round(n * rate)))
