# Engram Memory Characterization Eval

Characterizes Engram's memory system across six trials, measuring:
- Recall by query type (direct, paraphrase, adversarial decoy)
- RTRL novelty signal decay with repetition
- Forgetting curve after distractor interference
- Cold-session retrieval from episodic/semantic/cold storage
- Contradiction/fact-update resolution accuracy

## Setup

```bash
pip install aiohttp
export ANTHROPIC_API_KEY=sk-...
```

Wire `engram_probe.py` to your Engram instance:
- Set `ENGRAM_MODE = "http"` and adjust `base_url` endpoints, OR
- Set `ENGRAM_MODE = "direct"` and fill in the direct import block.

## Usage

```bash
# Full run (generates corpus, runs all trials, computes metrics)
python run_eval.py --output eval_results/ --corpus-cache eval_corpus.json

# Re-run specific trials (corpus already cached)
python run_eval.py --trials 4,5 --corpus-cache eval_corpus.json

# Recompute metrics from existing results
python run_eval.py --metrics-only --output eval_results/
```

## Trial Schedule

| # | Label | Subset | Reps | Distractors | Purpose |
|---|-------|--------|------|-------------|---------|
| 0 | baseline | all (60) | 1 | 0 | Cold recall baseline |
| 1 | reinforce_3x | high_salience (20) | 3 | 2 | Recall vs. repetition |
| 2 | reinforce_8x | high_salience (20) | 5 | 2 | Continued reinforcement |
| 3 | contradict | contradictions (20) | 1 | 1 | Fact update / overwrite |
| 4 | forgetting | none | 0 | 40 | Forgetting curve |
| 5 | cold_query | none | 0 | 0 | Cross-session retrieval |

## Output Files

```
eval_results/
  trial_00_baseline.json       # Raw injection + judgment results per trial
  trial_01_reinforce_3x.json
  ...
  all_trials.json              # All trials concatenated
  metrics.json                 # Computed learning curves + forgetting indices
```

## Key Metrics

- **recall_direct / recall_paraphrase**: Fraction of facts correctly retrieved
- **recall_decoy**: Fraction where adversarial query did NOT return wrong fact (precision proxy)
- **contamination_rate**: Fraction of retrievals with clearly wrong facts
- **novelty_mean**: Mean RTRL novelty signal during injection (should decay with repetitions)
- **forgetting_index**: recall_forgetting / recall_baseline (1.0 = no forgetting)
- **cold_query_index**: recall_cold / recall_baseline (tests persistent storage)

## Adapting to Engram's API

The three integration points in `engram_probe.py`:
1. `_http_inject()` — POST fact content to Engram's conversation endpoint
2. `_http_retrieve()` — POST query to Engram's retrieval endpoint
3. `reset_working_memory()` — Flush working memory layer for cold session simulation

If Engram exposes a novelty score (RTRL signal) in responses, include it in
the response JSON as `novelty_score`. The harness will track it across trials.
