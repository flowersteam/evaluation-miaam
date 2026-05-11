"""Async client for the MC-answer prediction task.

Asks the LLM to output a single integer = the option index the student will
pick on the next MC question. No logprobs, no constrained decoding — just
parse the integer from the response text.

Output is JSONL keyed by window_id, one row per window. Re-running with the
same --output overwrites.
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
from prompts_answer import build_messages  # noqa: E402

_INT_RE = re.compile(r"-?\d+")


def parse_choice_index(text: str, n_options: int) -> int | None:
    """Extract the first non-negative integer in [0, n_options) from text."""
    if not text:
        return None
    for m in _INT_RE.finditer(text):
        try:
            v = int(m.group(0))
        except ValueError:
            continue
        if 0 <= v < n_options:
            return v
    return None


def load_descriptions(path: Path) -> dict[str, str]:
    with open(path) as f:
        return json.load(f)


def load_expert_meta(exercises_table: Path) -> dict[str, str]:
    df = pl.read_parquet(exercises_table)
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
        for f in source_dir.iterdir():
            if f.suffix.lower() != ".png":
                continue
            with open(f, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            cache[(source_dir.name, f.stem)] = f"data:image/png;base64,{b64}"
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
    n_options = int(window["target_n_options"])
    messages = build_messages(
        history=window["history"],
        target={
            "exercise_id": window["target_exercise_id"],
            "source": window["target_source"],
            "n_options": n_options,
        },
        modality=args.modality,
        descriptions=descriptions,
        image_b64_cache=image_cache,
        expert_meta=expert_meta,
    )

    extra_body: dict = {}
    if args.reasoning == "off":
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if args.reasoning_effort:
        extra_body["reasoning"] = {"effort": args.reasoning_effort}
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
    pred = parse_choice_index(raw, n_options)
    truth = int(window["target_answer_idx"])
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
        "target_n_options": n_options,
        "target_answer_idx": truth,
        "target_correct": int(window["target_correct"]),
        "pred_answer_idx": pred,
        "pred_correct": int(pred == truth) if pred is not None else None,
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
    windows_df = pl.read_parquet(args.windows).sort("user_id_int")
    windows = windows_df.to_dicts()
    if args.max_windows:
        windows = windows[: args.max_windows]
    print(f"{len(windows)} windows to evaluate")
    if not windows:
        return

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
    n_unparsable = 0
    n_correct = 0
    t_start = time.perf_counter()
    with open(output, "w") as fout:
        async def run_and_write(w):
            nonlocal n_done, n_errors, n_unparsable, n_correct
            res = await evaluate_window(client, sem, w, descriptions, image_cache, expert_meta, args)
            fout.write(json.dumps(res) + "\n")
            fout.flush()
            n_done += 1
            if "error" in res:
                n_errors += 1
            elif res["pred_answer_idx"] is None:
                n_unparsable += 1
            elif res["pred_correct"]:
                n_correct += 1
            if n_done % args.log_every == 0:
                rate = n_done / (time.perf_counter() - t_start)
                acc = n_correct / max(n_done - n_errors - n_unparsable, 1)
                print(f"  {n_done}/{len(windows)} ({rate:.1f} win/s, "
                      f"errors={n_errors}, unparsable={n_unparsable}, acc={acc:.3f})")

        await asyncio.gather(*(run_and_write(w) for w in windows))

    elapsed = time.perf_counter() - t_start
    print(f"Done. {n_done} windows in {elapsed:.1f}s ({n_done/elapsed:.1f} win/s), "
          f"errors={n_errors}, unparsable={n_unparsable}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--windows", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--descriptions", required=True)
    p.add_argument("--screenshots-root", required=True)
    p.add_argument("--expert-knowledge", action="store_true")
    p.add_argument("--exercises-table", default=None)
    p.add_argument("--modality", choices=["text", "vision", "both"], default="text")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--model", default="google/gemma-4-31b-it")
    p.add_argument("--reasoning", choices=["off", "on"], default="on",
                   help="off → sends chat_template_kwargs={'enable_thinking': False} "
                        "(Qwen-style). Use 'on' (default) for backends that reject "
                        "that kwarg — OpenAI, Anthropic, Google via OpenRouter.")
    p.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None)
    p.add_argument("--max-tokens", type=int, default=64,
                   help="Headroom for hidden reasoning tokens + the integer answer. "
                        "OpenAI reasoning models enforce min 16. Bump to >=2048 for "
                        "Qwen3.6 with --reasoning on.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None,
                   help="Forwarded via extra_body. Qwen3.6 recommends 20.")
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
