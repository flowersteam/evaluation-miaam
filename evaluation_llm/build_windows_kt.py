"""Build evaluation windows from interactions_test.parquet.

We sample N target positions uniformly at random across the population of all
valid (student, target_idx) pairs. Each window carries the n preceding
attempts as history and the target attempt's ground-truth label.

Two filters are applied per student before windowing:

  1. **Listening exercises** (description contains "cout" — écouter/écoute/…)
     are dropped from the chronological sequence entirely, so they never
     appear in history and are never predicted on. They depend on audio
     that's not in the dataset.
  2. **Missing-asset windows** (history or target referencing an exercise
     without the required text description / screenshot) are dropped.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import polars as pl


def _load_descriptions(descriptions_path: Path) -> dict[str, str]:
    with open(descriptions_path) as f:
        return json.load(f)


def _listening_exercise_ids(descriptions: dict[str, str]) -> set[str]:
    """Exercises whose description contains 'cout' (case-insensitive).

    Catches the écouter / écoute / écoutes / Écouter family — listening
    exercises that depend on audio not present in the dataset. We exclude
    these from both target positions and history context: a student's
    listening attempts are pretended-not-to-have-happened when building
    windows.
    """
    return {eid for eid, desc in descriptions.items() if "cout" in desc.lower()}


def _load_vision_asset_set(screenshots_root: Path) -> set[tuple[str, str]]:
    """Returns set of (source, exercise_id) for which a compressed screenshot exists."""
    out: set[tuple[str, str]] = set()
    for source_dir in screenshots_root.iterdir():
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        for f in source_dir.iterdir():
            if f.suffix.lower() == ".png":
                out.add((source, f.stem))
    return out


def _required_assets_present(
    exercise_id: str,
    source: str,
    modality: str,
    text_ids: set[str],
    vision_ids: set[tuple[str, str]],
) -> bool:
    if modality in ("text", "both") and exercise_id not in text_ids:
        return False
    if modality in ("vision", "both") and (source, exercise_id) not in vision_ids:
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Absolute path to interactions_test.parquet")
    p.add_argument("--output", required=True,
                   help="Absolute path for the windows.parquet to write")
    p.add_argument("--descriptions", required=True,
                   help="Absolute path to Neurips/data/descriptions.json")
    p.add_argument("--screenshots-root", required=True,
                   help="Absolute path to Neurips/data/screenshots/compressed")
    p.add_argument("--n-history", type=int, default=20)
    p.add_argument("--min-history", type=int, default=None,
                   help="Default: same as --n-history.")
    p.add_argument("--n-windows", type=int, required=True,
                   help="Total number of windows to sample uniformly at random "
                        "across the population of valid (student, target) pairs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--require-modality", choices=["text", "vision", "both"], default="text")
    args = p.parse_args()

    n = args.n_history
    min_h = args.min_history if args.min_history is not None else n

    descriptions = _load_descriptions(Path(args.descriptions))
    text_ids = set(descriptions.keys())
    vision_ids = _load_vision_asset_set(Path(args.screenshots_root))
    listening_ids = _listening_exercise_ids(descriptions)
    print(f"Asset pools: {len(text_ids)} descriptions, {len(vision_ids)} screenshots")
    print(f"Listening (cout-) exercises excluded: {len(listening_ids)}")

    df = pl.read_parquet(args.input).sort(["user_id_int", "created_at"])
    n_attempts_total = len(df)
    print(f"Loaded {n_attempts_total} attempts across {df['user_id_int'].n_unique()} students")

    # Pass 1: collect filtered attempts per student and the flat list of all
    # candidate (user_id, target_idx) pairs. We hold attempts in memory once
    # so the second pass can hydrate the chosen windows cheaply.
    attempts_by_user: dict[int, list[dict]] = {}
    candidates: list[tuple[int, int]] = []
    n_students_seen = 0
    n_listening_attempts_dropped = 0

    for (user_id_int,), student_df in df.group_by(["user_id_int"], maintain_order=True):
        n_students_seen += 1
        attempts = student_df.to_dicts()
        before = len(attempts)
        attempts = [a for a in attempts if a["exercise_id"] not in listening_ids]
        n_listening_attempts_dropped += before - len(attempts)
        if len(attempts) < min_h + 1:
            continue
        attempts_by_user[int(user_id_int)] = attempts
        for tidx in range(min_h, len(attempts)):
            candidates.append((int(user_id_int), tidx))

    print(f"Candidate pool: {len(candidates)} (student, target) pairs across "
          f"{len(attempts_by_user)} students")

    rng = random.Random(args.seed)
    if args.n_windows < len(candidates):
        chosen = rng.sample(candidates, args.n_windows)
    else:
        print(f"--n-windows ({args.n_windows}) >= candidate pool size; keeping all.")
        chosen = list(candidates)
    chosen.sort()  # cluster by user_id so windows from the same student dispatch together

    rows = []
    n_windows_dropped_assets = 0

    # Pass 2: hydrate selected windows
    for user_id_int, target_idx in chosen:
        attempts = attempts_by_user[user_id_int]
        history_attempts = attempts[target_idx - n: target_idx]
        target = attempts[target_idx]

        assets_ok = all(
            _required_assets_present(a["exercise_id"], a["source"], args.require_modality,
                                     text_ids, vision_ids)
            for a in history_attempts + [target]
        )
        if not assets_ok:
            n_windows_dropped_assets += 1
            continue

        # data_duration is in milliseconds in the source parquet
        history_payload = [
            {
                "exercise_id": a["exercise_id"],
                "source": a["source"],
                "correct": int(a["data_correct"]),
                "duration_s": (int(a["data_duration"]) // 1000) if a["data_duration"] is not None else 0,
            }
            for a in history_attempts
        ]
        rows.append({
            "window_id": f"{user_id_int}:{target_idx}",
            "user_id_int": user_id_int,
            "target_idx": target_idx,
            "target_exercise_id": target["exercise_id"],
            "target_source": target["source"],
            "target_objective_id": target.get("objective_id"),
            "target_label": int(target["data_correct"]),
            "history": history_payload,
        })

    out_df = pl.DataFrame(rows)
    out_df.write_parquet(args.output)

    print(f"Students seen:        {n_students_seen}")
    print(f"Students contributing:{len({r['user_id_int'] for r in rows})}")
    print(f"Windows requested:    {args.n_windows}")
    print(f"Windows kept:         {len(out_df)}")
    print(f"Windows dropped (missing {args.require_modality} asset): {n_windows_dropped_assets}")
    print(f"Listening attempts dropped before windowing: {n_listening_attempts_dropped} "
          f"({n_listening_attempts_dropped / max(n_attempts_total, 1):.1%})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
