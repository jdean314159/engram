"""
Engram Memory Eval — Metrics

Computes per-trial and cross-trial statistics from trial records.
Outputs structured metrics dict and prints summary tables.
"""
import json
import math
from pathlib import Path
from typing import Optional


def _safe_mean(values: list) -> Optional[float]:
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def _recall_at_k(judgments: list[dict], k: int, query_type: str) -> float:
    """
    Recall@k: fraction of facts where top-k chunks contain the target.
    For decoy, 'correct' means NOT retrieved.
    """
    subset = [j for j in judgments if j["query_type"] == query_type]
    if not subset:
        return float("nan")
    return sum(1 for j in subset if j["correct"]) / len(subset)


def compute_trial_metrics(trial: dict, top_k_values: list[int]) -> dict:
    judgments = trial["judgment_results"]
    injection_results = trial["injection_results"]

    # Novelty scores from injections (RTRL signal) — exclude distractors
    novelty_scores = [
        r["novelty_score"]
        for r in injection_results
        if r.get("novelty_score") is not None
        and not str(r.get("fact_id", "")).startswith("distractor")
    ]
    # novelty_ratio = raw/EMA; more informative than raw as it's baseline-relative
    novelty_ratios = [
        r["novelty_ratio"]
        for r in injection_results
        if r.get("novelty_ratio") is not None
        and not str(r.get("fact_id", "")).startswith("distractor")
    ]

    metrics = {
        "trial_index": trial["trial_index"],
        "label": trial["label"],
        "subset": trial["subset_name"],
        "n_injections": len(injection_results),
        "n_judgments": len(judgments),
        "elapsed_sec": trial["elapsed_sec"],
    }

    for qt in ["direct", "paraphrase", "decoy"]:
        subset = [j for j in judgments if j["query_type"] == qt]
        metrics[f"recall_{qt}"] = (
            sum(1 for j in subset if j["correct"]) / len(subset)
            if subset else float("nan")
        )
        metrics[f"relevance_{qt}"] = _safe_mean([j["relevance"] for j in subset])

    # --- Contamination breakdown ---
    # Type 1 (decoy double-count): contaminated on a decoy query where retrieved=True
    #   → already captured by recall_decoy; excluded here to avoid double-counting
    # Type 2 (contradiction bleed): non-decoy query returns wrong/contradicting version
    #   → the real Engram architectural gap
    # Type 3 (name collision): corpus design flaw in decoy generation; not tracked separately
    non_decoy = [j for j in judgments if j["query_type"] != "decoy"]
    contradiction_bleed = [
        j for j in non_decoy
        if j["contaminated"] and not j["retrieved"]
    ]
    metrics["contradiction_bleed_rate"] = (
        len(contradiction_bleed) / len(non_decoy)
        if non_decoy else float("nan")
    )
    # Legacy metric kept for reference but now understood to be inflated by decoy double-counting
    metrics["contamination_rate"] = (
        sum(1 for j in judgments if j["contaminated"]) / len(judgments)
        if judgments else float("nan")
    )
    metrics["verbatim_match_rate"] = (
        sum(1 for j in judgments if j["verbatim_match"]) / len(judgments)
        if judgments else float("nan")
    )
    metrics["novelty_mean"] = _safe_mean(novelty_scores)
    metrics["novelty_min"] = min(novelty_scores) if novelty_scores else None
    metrics["novelty_max"] = max(novelty_scores) if novelty_scores else None
    # ratio = raw/EMA: >1.0 = more novel than baseline, <1.0 = familiar
    metrics["novelty_ratio_mean"] = _safe_mean(novelty_ratios)
    metrics["novelty_ratio_min"] = min(novelty_ratios) if novelty_ratios else None

    return metrics


def compute_learning_curves(all_trials: list[dict]) -> dict:
    """
    Cross-trial analysis:
    - How recall changes with cumulative injection count (learning curve)
    - How novelty signal decays with repetitions (novelty decay curve)
    - Forgetting index: recall_forgetting / recall_baseline
    """
    trial_metrics = []
    for t in all_trials:
        if t["judgment_results"]:
            trial_metrics.append(compute_trial_metrics(t, [1, 3, 5]))

    # Baseline (trial 0)
    baseline = next((m for m in trial_metrics if m["trial_index"] == 0), None)

    curves = {
        "per_trial": trial_metrics,
        "learning_curve": {
            "direct_recall": [m.get("recall_direct") for m in trial_metrics],
            "paraphrase_recall": [m.get("recall_paraphrase") for m in trial_metrics],
            "labels": [m["label"] for m in trial_metrics],
        },
        "novelty_decay": {
            "means": [m.get("novelty_mean") for m in trial_metrics],
            "labels": [m["label"] for m in trial_metrics],
        },
    }

    if baseline:
        forgetting_trial = next(
            (m for m in trial_metrics if m["label"] == "forgetting"), None
        )
        cold_trial = next(
            (m for m in trial_metrics if m["label"] == "cold_query"), None
        )

        def forgetting_index(trial_m, baseline_m, key):
            b = baseline_m.get(key)
            t = trial_m.get(key) if trial_m else None
            if b and t is not None and b > 0:
                return t / b
            return None

        curves["forgetting_index"] = {
            "direct_recall": forgetting_index(forgetting_trial, baseline, "recall_direct"),
            "paraphrase_recall": forgetting_index(forgetting_trial, baseline, "recall_paraphrase"),
        }
        curves["cold_query_index"] = {
            "direct_recall": forgetting_index(cold_trial, baseline, "recall_direct"),
            "paraphrase_recall": forgetting_index(cold_trial, baseline, "recall_paraphrase"),
        }

    return curves


def print_metrics_table(all_metrics: list[dict]):
    header = (
        f"{'Trial':<4} {'Label':<16} {'Recall(D)':<11} "
        f"{'Recall(P)':<11} {'Decoy✓':<8} "
        f"{'Contradiction%':<15} {'Novelty':<10} {'Secs':<6}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        def fmt(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "  N/A"
            return f"{v:>6.2%}" if isinstance(v, float) and v <= 1.0 else f"{v:>6.3f}"

        novelty = m.get("novelty_mean")
        novelty_str = f"{novelty:.4f}" if novelty is not None else "   N/A"
        print(
            f"{m['trial_index']:<4} {m['label']:<16} "
            f"{fmt(m.get('recall_direct')):<11} "
            f"{fmt(m.get('recall_paraphrase')):<11} "
            f"{fmt(m.get('recall_decoy')):<8} "
            f"{fmt(m.get('contradiction_bleed_rate')):<15} "
            f"{novelty_str:<10} "
            f"{m.get('elapsed_sec', 0):.1f}"
        )
    print("=" * len(header))


def save_metrics(all_trials: list[dict], output_dir: str):
    curves = compute_learning_curves(all_trials)
    out = Path(output_dir)
    (out / "metrics.json").write_text(json.dumps(curves, indent=2))
    print_metrics_table(curves["per_trial"])
    return curves
