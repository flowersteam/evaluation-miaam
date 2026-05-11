"""Async client for the correctness-prediction task.

Asks the LLM to emit a probability in [0, 1] for whether the next attempt
will be correct, then parses the float from the response. No logprobs, no
constrained decoding — works against any OpenAI-compatible backend
(OpenRouter, OpenAI, ...).

Output is JSONL keyed by window_id, one row per window. Re-running with
the same --output overwrites.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import polars as pl
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent))
from prompts_kt import build_messages  # noqa: E402

_FLOAT_RE = re.compile(r"[-+]?(?:\d+[.,]\d*|[.,]\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_probability(text: str) -> float | None:
    """Extract a probability in [0, 1] from a free-text response.

    Scans every numeric substring (accepting both `.` and French `,` as the
    decimal separator) and returns the first one that falls in [0, 1].
    Tolerates prose / reasoning leak before the answer ("Le 12 billes …
    donc 0,73") by walking past out-of-range matches like 12. Returns None
    only if the text contains no number in [0, 1] at all.
    """
    if not text:
        return None
    for m in _FLOAT_RE.finditer(text):
        try:
            v = float(m.group(0).replace(",", "."))
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
    return None


def load_descriptions(path: Path) -> dict[str, str]:
    with open(path) as f:
        return json.load(f)


def load_expert_meta(exercises_table: Path) -> dict[str, str]:
    """Build {exercise_id: formatted preamble} from maths_exercises_table.parquet.

    The preamble combines activity / objective names and pedagogical intents
    into a 2-line block prepended to each exercise in the prompt.
    """
    df = pl.read_parquet(exercises_table)
    needed = {"exercise_id", "objective_name", "objective_pedagogical_intent",
              "activity_name", "activity_pedagogical_intent"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{exercises_table} is missing required columns for --expert-knowledge: {missing}"
        )
    out: dict[str, str] = {}
    for r in df.iter_rows(named=True):
        eid = r["exercise_id"]
        an, ai = r.get("activity_name") or "", r.get("activity_pedagogical_intent") or ""
        on, oi = r.get("objective_name") or "", r.get("objective_pedagogical_intent") or ""
        lines = []
        if an or ai:
            lines.append(f"Activity ({an}): {ai}".strip())
        if on or oi:
            lines.append(f"Objective ({on}): {oi}".strip())
        if lines:
            out[eid] = "\n".join(lines)
    return out


def load_image_cache(screenshots_root: Path) -> dict[tuple[str, str], str]:
    cache: dict[tuple[str, str], str] = {}
    for source_dir in screenshots_root.iterdir():
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        for f in source_dir.iterdir():
            if f.suffix.lower() != ".png":
                continue
            with open(f, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            cache[(source, f.stem)] = f"data:image/png;base64,{b64}"
    return cache


async def evaluate_window(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    window: dict,
    descriptions: dict[str, str],
    image_cache: dict[tuple[str, str], str],
    expert_meta: dict[str, str] | None,
    args: argparse.Namespace,
) -> dict:
    messages = build_messages(
        history=window["history"],
        target={
            "exercise_id": window["target_exercise_id"],
            "source": window["target_source"],
        },
        modality=args.modality,
        descriptions=descriptions,
        image_b64_cache=image_cache,
        expert_meta=expert_meta,
    )

    extra_body: dict = {}
    if args.reasoning == "off":
        # Disables Qwen-style default thinking via chat_template_kwargs. Some
        # OpenAI-compatible providers (notably OpenAI / Anthropic / Google
        # via OpenRouter) reject this kwarg — pass --reasoning on for those.
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if args.reasoning_effort:
        extra_body["reasoning"] = {"effort": args.reasoning_effort}
    # Qwen3.6 non-thinking mode degenerates at temperature=0 — use top_k/
    # presence_penalty per the model card. Pass via extra_body since they
    # aren't in the OpenAI standard.
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k

    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
                max_tokens=args.max_tokens,
                extra_body=extra_body,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)
        except Exception as e:
            return {
                "window_id": window["window_id"],
                "error": f"{type(e).__name__}: {e}",
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            }

    choice = resp.choices[0]
    raw = choice.message.content or ""
    p = parse_probability(raw)
    n_reasoning = 0
    rt = getattr(choice.message, "reasoning_content", None)
    if rt:
        n_reasoning = len(rt)

    return {
        "window_id": window["window_id"],
        "user_id_int": window["user_id_int"],
        "target_idx": window["target_idx"],
        "target_exercise_id": window["target_exercise_id"],
        "target_source": window["target_source"],
        "target_objective_id": window.get("target_objective_id"),
        "target_label": int(window["target_label"]),
        "p_correct": p,
        "pred_label": int(p >= 0.5) if p is not None else None,
        "raw_answer": raw,
        "latency_ms": latency_ms,
        "model": args.model,
        "modality": args.modality,
        "reasoning_chars": n_reasoning,
    }


async def amain(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        print(f"Overwriting existing {output}")
        output.unlink()

    print("Loading descriptions and images…")
    descriptions = load_descriptions(Path(args.descriptions))
    image_cache: dict[tuple[str, str], str] = {}
    if args.modality in ("vision", "both"):
        image_cache = load_image_cache(Path(args.screenshots_root))
        print(f"Loaded {len(image_cache)} images into base64 cache")
    expert_meta: dict[str, str] | None = None
    if args.expert_knowledge:
        if not args.exercises_table:
            raise SystemExit("--expert-knowledge requires --exercises-table")
        expert_meta = load_expert_meta(Path(args.exercises_table))
        print(f"Loaded expert metadata for {len(expert_meta)} exercises")

    print(f"Loading windows from {args.windows}…")
    windows_df = pl.read_parquet(args.windows).sort("user_id_int")  # cluster per student
    windows = windows_df.to_dicts()
    if args.max_windows:
        windows = windows[: args.max_windows]
    print(f"{len(windows)} windows to evaluate")
    if not windows:
        return

    # max_retries=8 absorbs occasional 5xx / connection errors from the
    # provider. Vision prefill at large n_history can take minutes per
    # request — bump the per-request timeout when images are in the pipeline.
    request_timeout = 600.0 if args.modality != "text" else 120.0
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        max_retries=8,
        timeout=request_timeout,
    )
    print(f"OpenAI client: timeout={request_timeout}s, max_retries=8, concurrency={args.concurrency}")
    sem = asyncio.Semaphore(args.concurrency)

    output.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_errors = 0
    t_start = time.perf_counter()
    with open(output, "w") as fout:
        async def run_and_write(w):
            nonlocal n_done, n_errors
            res = await evaluate_window(client, sem, w, descriptions, image_cache, expert_meta, args)
            fout.write(json.dumps(res) + "\n")
            fout.flush()
            n_done += 1
            if "error" in res:
                n_errors += 1
            if n_done % args.log_every == 0:
                rate = n_done / (time.perf_counter() - t_start)
                print(f"  {n_done}/{len(windows)} ({rate:.1f} win/s, {n_errors} errors)")

        await asyncio.gather(*(run_and_write(w) for w in windows))

    elapsed = time.perf_counter() - t_start
    print(f"Done. {n_done} windows in {elapsed:.1f}s ({n_done/elapsed:.1f} win/s), {n_errors} errors")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--windows", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--descriptions", required=True)
    p.add_argument("--screenshots-root", required=True)
    p.add_argument("--expert-knowledge", action="store_true",
                   help="Augment each exercise's description with the activity / objective "
                        "names and pedagogical intents from the exercises table.")
    p.add_argument("--exercises-table", default=None,
                   help="Required when --expert-knowledge is set.")
    p.add_argument("--modality", choices=["text", "vision", "both"], default="text")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--model", default="google/gemma-4-31b-it")
    p.add_argument("--reasoning", choices=["off", "on"], default="on",
                   help="off → sends chat_template_kwargs={'enable_thinking': False} "
                        "(Qwen-style). Use 'on' (default) for backends that reject "
                        "that kwarg — OpenAI, Anthropic, Google via OpenRouter.")
    p.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None,
                   help="OpenRouter/OpenAI reasoning-effort knob. Only forwarded when set.")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="Headroom for hidden reasoning tokens + the float answer. "
                        "OpenAI reasoning models enforce min 16. Bump to >=2048 "
                        "when running Qwen3.6 with --reasoning on (it thinks).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 (greedy) is fine for OpenAI reasoning models; Qwen3.6 in "
                        "non-thinking mode degenerates at 0 — use 0.7 per the model card.")
    p.add_argument("--top-p", type=float, default=1.0,
                   help="Qwen3.6 non-thinking: 0.8. Qwen3.6 thinking: 0.95.")
    p.add_argument("--top-k", type=int, default=None,
                   help="Forwarded via extra_body. Qwen3.6 recommends 20.")
    p.add_argument("--presence-penalty", type=float, default=0.0,
                   help="Qwen3.6 non-thinking recommends 1.5 to avoid repetition loops.")
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
