"""
Inspect contaminated judgments from eval results, broken down by failure type.

Type 1 — Decoy precision failure: contaminated=True, retrieved=True, query_type=decoy
           The target fact was returned for an adversarial query.
           Already captured by Decoy✓ metric — not a new signal.

Type 2 — Contradiction bleed: contaminated=True, retrieved=False, query_type != decoy
           A contradicting version of the fact won retrieval over the original.
           This is the real Engram architectural gap.

Type 3 — Name collision (corpus flaw): decoy query too similar to a real fact
           e.g. "Golden Oak" vs "Grand Oak". Corpus generation quality issue.

Usage:
    python inspect_contaminated.py tests/eval/results/ [--type 1|2|3] [--trial 0]
"""
import json
import sys
import argparse
from pathlib import Path


def classify(j: dict) -> int:
    """Classify a contaminated judgment into failure type 1, 2, or 3."""
    is_decoy = j["query_type"] == "decoy"
    retrieved = j["retrieved"]

    if not is_decoy and not retrieved:
        return 2  # contradiction bleed

    if is_decoy and retrieved:
        # High relevance on a decoy = corpus name collision (type 3)
        # Low relevance = genuine precision failure (type 1)
        return 3 if j.get("relevance", 0) >= 3 else 1

    return 1  # catch-all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="tests/eval/results")
    parser.add_argument("--type", type=int, choices=[1, 2, 3], default=None,
                        help="Show only this failure type")
    parser.add_argument("--trial", type=int, default=None,
                        help="Show only this trial index")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_trials_path = results_dir / "all_trials.json"

    if not all_trials_path.exists():
        print(f"Not found: {all_trials_path}")
        sys.exit(1)

    trials = json.loads(all_trials_path.read_text())
    corpus_path = Path("tests/eval/eval_corpus.json")
    corpus = {f["id"]: f for f in json.loads(corpus_path.read_text())}

    type_counts = {1: 0, 2: 0, 3: 0}
    paradox_count = 0

    for trial in trials:
        if args.trial is not None and trial["trial_index"] != args.trial:
            continue

        label = trial["label"]
        contaminated = [j for j in trial["judgment_results"] if j["contaminated"]]
        paradox_count += sum(1 for j in contaminated if j["retrieved"])

        typed = [(j, classify(j)) for j in contaminated]
        for _, t in typed:
            type_counts[t] += 1

        if args.type is not None:
            typed = [(j, t) for j, t in typed if t == args.type]

        if not typed:
            continue

        print(f"\n{'='*70}")
        print(f"Trial {trial['trial_index']}: {label}  "
              f"({len(contaminated)} contaminated / {len(trial['judgment_results'])} total)")
        counts = {t: sum(1 for _, x in typed if x == t) for t in [1, 2, 3]}
        print(f"  T1(decoy-precision)={counts[1]}  "
              f"T2(contradiction-bleed)={counts[2]}  "
              f"T3(name-collision)={counts[3]}")
        print('='*70)

        for j, ftype in typed:
            fact_id = j["fact_id"]
            fact = corpus.get(fact_id, {})
            type_label = {
                1: "DECOY-PRECISION",
                2: "CONTRADICTION-BLEED",
                3: "NAME-COLLISION",
            }[ftype]

            print(f"\n  [{type_label}]")
            print(f"  fact_id    : {fact_id}")
            print(f"  query_type : {j['query_type']}")
            print(f"  retrieved  : {j['retrieved']}   relevance: {j['relevance']}/5")
            print(f"  notes      : {j['notes']}")
            print(f"  canonical  : {fact.get('canonical', 'N/A')}")
            if j["query_type"] == "decoy":
                print(f"  decoy_query: {fact.get('decoy_query', 'N/A')}")

    total = sum(type_counts.values())
    print(f"\n{'='*70}")
    print(f"SUMMARY (trials shown: "
          f"{'all' if args.trial is None else args.trial}, "
          f"type filter: {'all' if args.type is None else args.type})")
    print(f"  Total contaminated          : {total}")
    print(f"  Type 1 — Decoy precision    : {type_counts[1]}"
          f"  (already in Decoy metric, not a new signal)")
    print(f"  Type 2 — Contradiction bleed: {type_counts[2]}"
          f"  (*** Engram bug: no conflict resolution ***)")
    print(f"  Type 3 — Name collision     : {type_counts[3]}"
          f"  (corpus flaw: tighten decoy generation prompt)")
    print(f"  Paradox (contam + retrieved): {paradox_count}")


if __name__ == "__main__":
    main()
