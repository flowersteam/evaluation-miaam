"""Prompt construction for the LLM/VLM correctness-prediction eval.

`build_messages` returns OpenAI chat-format messages (system + single user
message) so the same prompt works against any OpenAI-compatible endpoint
(OpenRouter, OpenAI, ...).

The user message is a list of content blocks, ordered chronologically by
attempt and ending with the target. The static system prefix and the
per-student history prefix help server-side prefix caching when windows
from the same student are dispatched consecutively.
"""

from __future__ import annotations

from typing import Iterable, Literal

Modality = Literal["text", "vision", "both"]

SYSTEM_PROMPT = (
    "You are a knowledge-tracing model for French primary and secondary school "
    "mathematics. You will see a chronological sequence of past exercise attempts "
    "from a single student, each annotated with the outcome (correct/incorrect). "
    "You must then estimate the probability that the student "
    "will answer the next exercise correctly. "
    "Reply with a single number between 0 and 1 (e.g. 0.73). Output only the "
    "number, no explanation, no other text."
)


def _exercise_block(
    exercise_id: str,
    source: str,
    modality: Modality,
    descriptions: dict[str, str],
    image_b64_cache: dict[tuple[str, str], str],
    expert_meta: dict[str, str] | None = None,
) -> list[dict]:
    """Content blocks for a single exercise (text, image, or both).

    If `expert_meta` is provided, prepend the formatted expert preamble
    (objective + activity names and pedagogical intents) before the
    description / screenshot.
    """
    blocks: list[dict] = []
    if expert_meta is not None:
        ek = expert_meta.get(exercise_id)
        if ek:
            blocks.append({"type": "text", "text": ek})
    if modality in ("text", "both"):
        desc = descriptions.get(exercise_id, "[description unavailable]")
        blocks.append({"type": "text", "text": desc})
    if modality in ("vision", "both"):
        data_uri = image_b64_cache.get((source, exercise_id))
        if data_uri is None:
            blocks.append({"type": "text", "text": "[screenshot unavailable]"})
        else:
            blocks.append({"type": "image_url", "image_url": {"url": data_uri}})
    return blocks


def build_messages(
    history: Iterable[dict],
    target: dict,
    modality: Modality,
    descriptions: dict[str, str],
    image_b64_cache: dict[tuple[str, str], str],
    expert_meta: dict[str, str] | None = None,
) -> list[dict]:
    """Build the chat messages for one window.

    `history` is an iterable of dicts with keys
        exercise_id, source, correct (0|1), duration_s
    in chronological order. `target` is a dict with keys
        exercise_id, source. `expert_meta`, when provided, maps
        exercise_id to a formatted multi-line preamble string.
    """
    user_blocks: list[dict] = []
    history = list(history)
    n = len(history)
    for i, h in enumerate(history, start=1):
        user_blocks.append({"type": "text", "text": f"Attempt {i}/{n}:"})
        user_blocks.extend(
            _exercise_block(h["exercise_id"], h["source"], modality,
                            descriptions, image_b64_cache, expert_meta)
        )
        outcome = "correct" if int(h["correct"]) == 1 else "incorrect"
        user_blocks.append({"type": "text", "text": f"-> Outcome: {outcome}"})

    user_blocks.append({"type": "text", "text": "Next attempt:"})
    user_blocks.extend(
        _exercise_block(target["exercise_id"], target["source"], modality,
                        descriptions, image_b64_cache, expert_meta)
    )
    user_blocks.append(
        {
            "type": "text",
            "text": "What is the probability the student will get this correct? "
                    "Reply with a single number between 0 and 1 (e.g. 0.73). Output only the number.",
        }
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_blocks},
    ]
