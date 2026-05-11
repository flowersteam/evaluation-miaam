"""Build evaluation windows for the MC-answer prediction task.

Difference from build_windows.py:

* We restrict to gameplay_type == MULTIPLE_CHOICE. Non-MC interactions are
  dropped from each student's chronological sequence (so the sequence
  contracts) but the per-student ordering of remaining MC attempts is
  preserved. A student with mixed history contributes only their MC
  attempts, in chronological order.
* The ground truth is the student's chosen option index (`data_answer`),
  not the binary correctness. We keep `data_correct` in history payloads
  so the model knows whether the student's prior pick was right.
* We compute N_options per exercise empirically from the union of all
  observed answer indices in the source parquet (max index + 1). This
  bounds the model's answer space at evaluation time.

We drop:
  - rows whose data_answer is empty (`[]`) or has >1 element
    (multi-select MC, ~13 rows out of ~400k — too rare to model)
  - exercises with empirical N_options < 2 (no choice to predict)
  - listening (cout-) exercises (audio dependency)
  - windows where any history/target asset is missing for the requested modality
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import polars as pl


def _load_descriptions(path: Path) -> dict[str, str]:
    with open(path) as f:
        return json.load(f)


def _listening_exercise_ids(descriptions: dict[str, str]) -> set[str]:
    return {eid for eid, desc in descriptions.items() if "cout" in desc.lower()}


def _load_vision_asset_set(screenshots_root: Path) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for source_dir in screenshots_root.iterdir():
        if not source_dir.is_dir():
            continue
        for f in source_dir.iterdir():
            if f.suffix.lower() == ".png":
                out.add((source_dir.name, f.stem))
    return out


def _required_assets_present(
    exercise_id: str, source: str, modality: str,
    text_ids: set[str], vision_ids: set[tuple[str, str]],
) -> bool:
    if modality in ("text", "both") and exercise_id not in text_ids:
        return False
    if modality in ("vision", "both") and (source, exercise_id) not in vision_ids:
        return False
    return True


def _parse_single_answer(s: str | None) -> int | None:
    """Return the picked index for single-pick MC, or None if invalid/multi-select."""
    if s is None:
        return None
    try:
        v = json.loads(s)
    except (TypeError, ValueError):
        return None
    if not isinstance(v, list) or len(v) != 1:
        return None
    pick = v[0]
    if not isinstance(pick, int) or pick < 0:
        return None
    return pick


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Absolute path to interactions_test.parquet")
    p.add_argument("--exercises-table", required=True,
                   help="Absolute path to maths_exercises_table.parquet "
                        "(used to filter gameplay_type==MULTIPLE_CHOICE)")
    p.add_argument("--output", required=True,
                   help="Absolute path for the windows.parquet to write")
    p.add_argument("--descriptions", required=True,
                   help="Absolute path to Neurips/data/descriptions.json")
    p.add_argument("--screenshots-root", required=True,
                   help="Absolute path to Neurips/data/screenshots/compressed")
    p.add_argument("--n-history", type=int, default=20)
    p.add_argument("--min-history", type=int, default=None,
                   help="Default: same as --n-history.")
    p.add_argument("--n-windows", type=int, required=True)
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

    ex_df = pl.read_parquet(args.exercises_table, columns=["exercise_id", "gameplay_type"])
    mc_ids = set(ex_df.filter(pl.col("gameplay_type") == "MULTIPLE_CHOICE")["exercise_id"].to_list())
    print(f"MULTIPLE_CHOICE exercises in table: {len(mc_ids)}")

    df = pl.read_parquet(args.input).sort(["user_id_int", "created_at"])
    n_total = len(df)
    df = df.filter(pl.col("exercise_id").is_in(list(mc_ids)))
    print(f"Filtered to MC: {len(df)}/{n_total} attempts ({100*len(df)/max(n_total,1):.1f}%)")

    # Decode data_answer once, drop invalid/multi-select rows.
    df = df.with_columns(
        pl.col("data_answer").map_elements(_parse_single_answer, return_dtype=pl.Int64).alias("answer_idx")
    ).filter(pl.col("answer_idx").is_not_null())
    print(f"After dropping empty / multi-select answers: {len(df)} attempts")

    # Empirical N_options per exercise (max picked index + 1). Drop exercises
    # where this is <2 — no actual choice to predict.
    n_opts = (
        df.group_by("exercise_id")
          .agg((pl.col("answer_idx").max() + 1).alias("n_options"))
    )
    keep_ids = set(n_opts.filter(pl.col("n_options") >= 2)["exercise_id"].to_list())
    df = df.filter(pl.col("exercise_id").is_in(list(keep_ids)))
    n_opts_map = dict(n_opts.filter(pl.col("exercise_id").is_in(list(keep_ids)))
                            .iter_rows())
    print(f"After dropping exercises with <2 observed options: {len(df)} attempts, "
          f"{len(keep_ids)} exercises")

    # Pass 1: per-student MC sequence + candidate (user, target_idx) pairs.
    attempts_by_user: dict[int, list[dict]] = {}
    candidates: list[tuple[int, int]] = []
    n_listening_dropped = 0
    for (user_id_int,), student_df in df.group_by(["user_id_int"], maintain_order=True):
        attempts = student_df.to_dicts()
        before = len(attempts)
        attempts = [a for a in attempts if a["exercise_id"] not in listening_ids]
        n_listening_dropped += before - len(attempts)
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
    n_dropped_assets = 0
    for user_id_int, target_idx in chosen:
        attempts = attempts_by_user[user_id_int]
        history = attempts[target_idx - n: target_idx]
        target = attempts[target_idx]
        if not all(_required_assets_present(a["exercise_id"], a["source"],
                                            args.require_modality, text_ids, vision_ids)
                   for a in history + [target]):
            n_dropped_assets += 1
            continue
        history_payload = [
            {
                "exercise_id": a["exercise_id"],
                "source": a["source"],
                "answer_idx": int(a["answer_idx"]),
                "correct": int(a["data_correct"]),
                "duration_s": (int(a["data_duration"]) // 1000) if a["data_duration"] is not None else 0,
            }
            for a in history
        ]
        rows.append({
            "window_id": f"{user_id_int}:{target_idx}",
            "user_id_int": user_id_int,
            "target_idx": target_idx,
            "target_exercise_id": target["exercise_id"],
            "target_source": target["source"],
            "target_objective_id": target.get("objective_id"),
            "target_n_options": int(n_opts_map[target["exercise_id"]]),
            "target_answer_idx": int(target["answer_idx"]),
            "target_correct": int(target["data_correct"]),
            "history": history_payload,
        })

    out_df = pl.DataFrame(rows)
    out_df.write_parquet(args.output)

    print(f"Students contributing: {len({r['user_id_int'] for r in rows})}")
    print(f"Windows requested: {args.n_windows}")
    print(f"Windows kept:      {len(out_df)}")
    print(f"Windows dropped (missing {args.require_modality} asset): {n_dropped_assets}")
    print(f"Listening attempts dropped before windowing: {n_listening_dropped}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
