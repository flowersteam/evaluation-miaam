"""Score a results.jsonl produced by run_eval_kt.py.

Reports AUC, accuracy @ 0.5, Brier score, plus per-source (am/mia) and
per-objective breakdowns. Joins the target_objective_id from the windows
parquet if results.jsonl rows lack it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import polars as pl
from sklearn.metrics import roc_auc_score, brier_score_loss


def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def safe_auc(labels, probs):
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def summarize(rows: list[dict]) -> dict:
    n_total = len(rows)
    n_errors = sum(1 for r in rows if "error" in r)
    # Treat unparseable responses as p_correct=0.5 (a fully uncertain prediction).
    # This penalizes the model's failure-to-format only mildly: 0.5 contributes
    # 0.25 to Brier regardless of label and is on the AUC decision boundary.
    n_unparsed = 0
    valid = []
    for r in rows:
        if "error" in r:
            continue
        if r.get("p_correct") is None:
            r = {**r, "p_correct": 0.5}
            n_unparsed += 1
        valid.append(r)

    n_valid = len(valid)
    n_no_prob = n_unparsed  # kept under old name in returned dict for backwards compat

    if not valid:
        return {
            "n_total": n_total, "n_valid": 0, "n_errors": n_errors,
            "n_no_prob": n_no_prob,
        }

    labels = [int(r["target_label"]) for r in valid]
    probs = [float(r["p_correct"]) for r in valid]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    accuracy = sum(int(l == p) for l, p in zip(labels, preds)) / len(labels)
    brier = float(brier_score_loss(labels, probs))
    auc = safe_auc(labels, probs)

    # Per-label (AUC is undefined for a single class — accuracy here is recall
    # on label=1 and specificity on label=0)
    label_metrics = {}
    for lbl in (1, 0):
        rs = [r for r in valid if int(r["target_label"]) == lbl]
        if not rs:
            continue
        ps = [float(r["p_correct"]) for r in rs]
        n_correct = sum(1 for p in ps if (p >= 0.5) == bool(lbl))
        label_metrics[lbl] = {
            "n": len(rs),
            "accuracy": n_correct / len(rs),
            "mean_p": sum(ps) / len(rs),
        }

    # Per-source
    per_source = defaultdict(list)
    for r in valid:
        per_source[r["target_source"]].append(r)
    source_metrics = {}
    for src, rs in per_source.items():
        ls = [int(r["target_label"]) for r in rs]
        ps = [float(r["p_correct"]) for r in rs]
        source_metrics[src] = {
            "n": len(rs),
            "auc": safe_auc(ls, ps),
            "accuracy": sum(int(l == (1 if p >= 0.5 else 0)) for l, p in zip(ls, ps)) / len(rs),
            "brier": float(brier_score_loss(ls, ps)) if len(set(ls)) >= 1 else float("nan"),
        }

    # Per-objective (only if target_objective_id is present)
    per_obj = defaultdict(list)
    for r in valid:
        oid = r.get("target_objective_id")
        if oid:
            per_obj[oid].append(r)
    obj_metrics = {}
    for oid, rs in per_obj.items():
        if len(rs) < 20:  # not enough samples for meaningful AUC
            continue
        ls = [int(r["target_label"]) for r in rs]
        ps = [float(r["p_correct"]) for r in rs]
        obj_metrics[oid] = {
            "n": len(rs),
            "auc": safe_auc(ls, ps),
        }

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_errors": n_errors,
        "n_no_prob": n_no_prob,
        "auc": auc,
        "accuracy": accuracy,
        "brier": brier,
        "by_label": {str(k): v for k, v in label_metrics.items()},
        "by_source": source_metrics,
        "by_objective": obj_metrics,
    }


def maybe_join_objectives(rows: list[dict], windows_path: Path | None) -> list[dict]:
    """Backfill target_objective_id on result rows from windows.parquet if missing."""
    if not windows_path or not windows_path.exists():
        return rows
    if rows and rows[0].get("target_objective_id"):
        return rows
    wdf = pl.read_parquet(windows_path).select(["window_id", "target_objective_id"])
    lookup = {r["window_id"]: r["target_objective_id"] for r in wdf.to_dicts()}
    for r in rows:
        if not r.get("target_objective_id"):
            r["target_objective_id"] = lookup.get(r["window_id"])
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True,
                   help="Absolute path to results.jsonl")
    p.add_argument("--windows", required=True,
                   help="Absolute path to windows.parquet (used to backfill objective ids)")
    p.add_argument("--out", default=None, help="Write metrics JSON to this path")
    args = p.parse_args()

    rows = load_results(Path(args.results))
    rows = maybe_join_objectives(rows, Path(args.windows) if args.windows else None)
    summary = summarize(rows)

    print(f"Total rows:   {summary['n_total']}")
    print(f"Valid:        {summary['n_valid']}  (incl. {summary['n_no_prob']} backfilled to p=0.5)")
    print(f"Errors:       {summary['n_errors']}")
    if summary["n_valid"] == 0:
        return
    print()
    print(f"AUC:        {summary['auc']:.4f}")
    print(f"Accuracy:   {summary['accuracy']:.4f}")
    print(f"Brier:      {summary['brier']:.4f}")
    if summary.get("by_label"):
        print()
        print("By label (AUC undefined for single class):")
        for lbl in ("1", "0"):
            m = summary["by_label"].get(lbl)
            if m:
                print(f"  label={lbl}  n={m['n']:6d}  acc={m['accuracy']:.4f}  mean_p={m['mean_p']:.4f}")
    print()
    print("By source:")
    for src, m in summary["by_source"].items():
        print(f"  {src:6s} n={m['n']:6d}  auc={m['auc']:.4f}  acc={m['accuracy']:.4f}  brier={m['brier']:.4f}")
    if summary["by_objective"]:
        print()
        print(f"By objective (n>=20): {len(summary['by_objective'])} objectives")
        worst = sorted(summary["by_objective"].items(), key=lambda kv: kv[1]["auc"])[:5]
        best = sorted(summary["by_objective"].items(), key=lambda kv: -kv[1]["auc"])[:5]
        print("  Worst 5:")
        for oid, m in worst:
            print(f"    {oid}  n={m['n']:5d}  auc={m['auc']:.4f}")
        print("  Best 5:")
        for oid, m in best:
            print(f"    {oid}  n={m['n']:5d}  auc={m['auc']:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
