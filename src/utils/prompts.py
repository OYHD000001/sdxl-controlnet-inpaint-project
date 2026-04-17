from __future__ import annotations

import random


def maybe_drop_prompt(text: str, dropout: float) -> str:
    if dropout <= 0:
        return text
    if random.random() < dropout:
        return ""
    return text
