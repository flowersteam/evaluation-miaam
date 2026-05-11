"""Score MC answer-prediction results.

Top-1 accuracy of the model's predicted choice index against the student's
recorded pick, plus breakdowns by source/objective/n_options. Reports two
reference baselines per slice: random (1/n_options averaged) and "always
pick the correct answer" (i.e. accuracy if the model perfectly predicted
correctness — a weak student-model baseline that beats random when
students mostly answer correctly).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import polars as pl


def load_results(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _slice_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    n_correct = sum(1 for r in rows if r.get("pred_correct") == 1)
    avg_inv_n = sum(1.0 / max(int(r["target_n_options"]), 1) for r in rows) / n
    n_student_correct = sum(1 for r in rows if int(r.get("target_correct", 0)) == 1)
    return {
        "n": n,
        "accuracy": n_correct / n,
        "random_baseline": avg_inv_n,
        "correct_answer_baseline": n_student_correct / n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--windows", required=True,
                   help="windows.parquet (only used for sanity-counting)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = load_results(Path(args.results))
    n_total = len(rows)
    n_errors = sum(1 for r in rows if "error" in r)

    # Backfill unparseable predictions with a deterministic uniform random pick
    # in [0, n_options). This treats the model's failure-to-format as a
    # uniform-prior guess rather than dropping the row from the denominator.
    rng = random.Random(0)
    n_unparsable = 0
    valid: list[dict] = []
    for r in rows:
        if "error" in r:
            continue
        if r.get("pred_answer_idx") is None:
            n_opts = int(r["target_n_options"])
            pick = rng.randint(0, n_opts - 1)
            truth = int(r["target_answer_idx"])
            r = {**r, "pred_answer_idx": pick, "pred_correct": int(pick == truth)}
            n_unparsable += 1
        valid.append(r)

    print(f"Rows: {n_total}  errors: {n_errors}  "
          f"unparsable (filled w/ random): {n_unparsable}  scored: {len(valid)}")

    overall = _slice_metrics(valid)

    by_source: dict[str, list[dict]] = defaultdict(list)
    by_obj: dict[str, list[dict]] = defaultdict(list)
    by_nopts: dict[int, list[dict]] = defaultdict(list)
    for r in valid:
        by_source[str(r.get("target_source") or "?")].append(r)
        oid = r.get("target_objective_id")
        if oid is not None:
            by_obj[str(oid)].append(r)
        by_nopts[int(r["target_n_options"])].append(r)

    summary = {
        "n_total": n_total,
        "n_errors": n_errors,
        "n_unparsable": n_unparsable,
        "n_scored": len(valid),
        **overall,
        "by_source": {k: _slice_metrics(v) for k, v in by_source.items()},
        "by_n_options": {str(k): _slice_metrics(v) for k, v in sorted(by_nopts.items())},
        "by_objective": {k: _slice_metrics(v) for k, v in by_obj.items() if len(v) >= 20},
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Accuracy:                 {summary['accuracy']:.4f}")
    print(f"  random baseline:        {summary['random_baseline']:.4f}")
    print(f"  correct-answer baseln:  {summary['correct_answer_baseline']:.4f}")
    print()
    print("By n_options:")
    for k, m in summary["by_n_options"].items():
        print(f"  n_opts={k}  n={m['n']:5d}  acc={m['accuracy']:.4f}  "
              f"random={m['random_baseline']:.4f}  correct={m['correct_answer_baseline']:.4f}")
    print()
    print("By source:")
    for k, m in summary["by_source"].items():
        print(f"  {k:8s} n={m['n']:5d}  acc={m['accuracy']:.4f}  "
              f"random={m['random_baseline']:.4f}  correct={m['correct_answer_baseline']:.4f}")

    if summary["by_objective"]:
        ranked = sorted(summary["by_objective"].items(), key=lambda kv: kv[1]["accuracy"])
        print("\n5 hardest objectives (lowest accuracy):")
        for oid, m in ranked[:5]:
            print(f"  {oid}  n={m['n']:5d}  acc={m['accuracy']:.4f}")
        print("\n5 easiest objectives (highest accuracy):")
        for oid, m in ranked[-5:][::-1]:
            print(f"  {oid}  n={m['n']:5d}  acc={m['accuracy']:.4f}")

    # sanity: results-row count vs windows file
    try:
        nw = len(pl.read_parquet(args.windows))
        if nw != n_total:
            print(f"\n⚠ windows.parquet has {nw} rows, results.jsonl has {n_total}.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
