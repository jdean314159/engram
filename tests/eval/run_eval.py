"""
Engram Memory Eval — Entry Point

Usage:
  python run_eval.py [--config config.json] [--output eval_results/]
                     [--corpus-cache corpus.json] [--trials 0,1,2]
                     [--skip-generate]

Steps:
  1. Generate (or load) fact corpus via Claude
  2. Run trial schedule against Engram
  3. Compute metrics and print summary table
  4. Save all results to --output directory

Requires:
  - Engram running (adjust ENGRAM_MODE in engram_probe.py)
  - ANTHROPIC_API_KEY in environment (for fact gen + judge)
  - aiohttp: pip install aiohttp
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from eval_config import EvalConfig
from fact_generator import generate_corpus
from engram_probe import EngramProbe
from claude_judge import ClaudeJudge, OllamaJudge
from trial_runner import TrialRunner
from metrics import save_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Engram Memory Characterization Eval")
    p.add_argument("--config", default=None, help="Path to eval_config.json")
    p.add_argument("--output", default="eval_results", help="Output directory")
    p.add_argument("--corpus-cache", default="eval_corpus.json",
                   help="Cache file for generated facts (avoids re-generation)")
    p.add_argument("--trials", default=None,
                   help="Comma-separated trial indices to run (default: all)")
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip corpus generation (corpus-cache must exist)")
    p.add_argument("--metrics-only", action="store_true",
                   help="Only recompute metrics from existing all_trials.json")
    p.add_argument("--judge", default="claude",
                   choices=["claude", "ollama"],
                   help="Judge backend: claude (default) or ollama")
    p.add_argument("--judge-model", default="qwen3:32b",
                   help="Model for ollama judge (default: qwen3:32b)")
    p.add_argument("--ollama-url", default="http://localhost:11434",
                   help="Ollama base URL (default: http://localhost:11434)")
    return p.parse_args()


async def main():
    args = parse_args()

    # Config
    config = EvalConfig.load(args.config) if args.config else EvalConfig()
    config.output_dir = args.output

    # Metrics-only mode
    if args.metrics_only:
        all_trials_path = Path(args.output) / "all_trials.json"
        if not all_trials_path.exists():
            print(f"[error] {all_trials_path} not found", file=sys.stderr)
            sys.exit(1)
        all_trials = json.loads(all_trials_path.read_text())
        save_metrics(all_trials, args.output)
        return

    # Generate / load corpus
    corpus = await generate_corpus(
        n_per_category=config.facts_per_category,
        model=config.judge_model,
        cache_path=args.corpus_cache if not args.skip_generate else None,
    )
    print(f"[main] Corpus: {len(corpus)} facts loaded")

    # Filter trials if requested
    if args.trials:
        indices = {int(x) for x in args.trials.split(",")}
        config.schedule = [
            s for i, s in enumerate(config.schedule) if i in indices
        ]
        print(f"[main] Running trials: {[s[0] for s in config.schedule]}")

    # Initialize probe and judge
    probe = EngramProbe(config)
    if args.judge == "ollama":
        judge = OllamaJudge(
            model=args.judge_model,
            base_url=args.ollama_url,
            max_tokens=config.judge_max_tokens,
        )
        print(f"[main] Using Ollama judge: {args.judge_model} @ {args.ollama_url}")
    else:
        judge = ClaudeJudge(model=config.judge_model, max_tokens=config.judge_max_tokens)
        print(f"[main] Using Claude judge: {config.judge_model}")

    await probe.start()
    await judge.start()

    try:
        runner = TrialRunner(config, corpus, probe, judge)
        await runner.run_all(resume=True)

        # Compute and save metrics
        all_trials_path = Path(args.output) / "all_trials.json"
        all_trials = json.loads(all_trials_path.read_text())
        save_metrics(all_trials, args.output)

    finally:
        await probe.stop()
        await judge.stop()


if __name__ == "__main__":
    asyncio.run(main())
