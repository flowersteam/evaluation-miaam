"""Prompts for the MC answer-prediction task.

The model sees the student's MC history (each attempt = question content +
which option index the student picked + whether that pick was correct) and
must predict the option index the student will pick on the next question.

Options are referred to by 0-indexed integer in reading order (top-to-bottom,
left-to-right). The model is told the answer space size (N_options) for the
target so out-of-range predictions don't sneak in.
"""

from __future__ import annotations

from typing import Iterable, Literal

from prompts_kt import _exercise_block  # reuse the modality-aware content builder

Modality = Literal["text", "vision", "both"]

SYSTEM_PROMPT_ANSWER = (
    "You are a student-modeling assistant for French primary and secondary school "
    "mathematics. You see a chronological sequence of past multiple-choice attempts "
    "from a single student — each attempt shows the question and which option "
    "index the student picked, with the outcome (correct/incorrect). You must "
    "predict which option index THIS student will pick on the next question. "
    "Options are 0-indexed in reading order (top-to-bottom, left-to-right). "
    "Predict the student's behavior, not the correct answer — students often "
    "make characteristic mistakes. "
    "Reply with a single non-negative integer. Output only the number."
)


def build_messages(
    history: Iterable[dict],
    target: dict,
    modality: Modality,
    descriptions: dict[str, str],
    image_b64_cache: dict[tuple[str, str], str],
    expert_meta: dict[str, str] | None = None,
) -> list[dict]:
    """Build chat messages for one MC-answer-prediction window.

    `history` items: {exercise_id, source, answer_idx, correct, duration_s}
    `target` items:  {exercise_id, source, n_options}
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
        user_blocks.append({
            "type": "text",
            "text": f"-> Student picked option {int(h['answer_idx'])} ({outcome})",
        })

    user_blocks.append({"type": "text", "text": "Next attempt:"})
    user_blocks.extend(
        _exercise_block(target["exercise_id"], target["source"], modality,
                        descriptions, image_b64_cache, expert_meta)
    )
    n_options = int(target["n_options"])
    valid = ", ".join(str(i) for i in range(n_options))
    user_blocks.append({
        "type": "text",
        "text": (f"This question has {n_options} options indexed 0..{n_options - 1} "
                 f"(reading order). Which option will THIS student pick? "
                 f"Reply with one integer from {{{valid}}}. Output only the number."),
    })

    return [
        {"role": "system", "content": SYSTEM_PROMPT_ANSWER},
        {"role": "user", "content": user_blocks},
    ]
