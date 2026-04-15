"""
Engram Memory Eval — Configuration
"""
from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class EvalConfig:
    # --- Corpus ---
    n_facts: int = 60                   # Total facts generated
    n_categories: int = 6               # Categories: people/events/prefs/skills/contradictions/distractors
    facts_per_category: int = 10

    # --- Injection schedule ---
    # Each entry: (fact_subset_name, repetitions, distractor_episodes_before)
    # Fact subsets: "all", "high_salience", "contradictions"
    schedule: list = field(default_factory=lambda: [
        # (label,           subset,            reps, distractors_before)
        ("baseline",        "all",             1,    0),
        ("reinforce_3x",    "high_salience",   3,    2),
        ("reinforce_8x",    "high_salience",   5,    2),   # cumulative 8x
        ("contradict",      "contradictions",  1,    1),
        ("forgetting",      "none", 0, 200),   # was 40
        ("cold_query",      "none",            0,    0),   # query without inject
    ])

    # --- Retrieval ---
    retrieve_top_k: list = field(default_factory=lambda: [1])

    # --- Claude judge ---
    judge_model: str = "claude-sonnet-4-20250514"
    judge_max_tokens: int = 512

    # --- Engram connection ---
    # Adapt these to your actual Engram API surface
    engram_base_url: str = "http://localhost:8000"  # or direct import
    engram_project: str = "eval_harness"
    local_model: str = "qwen3:9b"                   # executor model via Ollama

    # --- Output ---
    output_dir: str = "eval_results"
    random_seed: int = 42

    def save(self, path: str):
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: str) -> "EvalConfig":
        d = json.loads(Path(path).read_text())
        return cls(**d)
