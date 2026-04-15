"""
Engram Memory Eval — Trial Runner

Executes the injection/query schedule and collects raw results.

Trial structure per schedule entry:
  1. Inject N distractors (interference)
  2. Inject target subset (possibly using contradictions)
  3. Retrieve all facts (direct + paraphrase + decoy queries)
  4. Judge all retrievals via Claude
  5. Record InjectionResult + JudgmentResult per fact per trial
"""
import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from eval_config import EvalConfig
from fact_generator import Fact, FactCorpus
from engram_probe import EngramProbe, InjectionResult
from claude_judge import ClaudeJudge, JudgmentResult


@dataclass
class TrialRecord:
    trial_index: int
    label: str
    subset_name: str
    repetitions: int
    distractors_injected: int
    injection_results: list[dict]
    judgment_results: list[dict]
    elapsed_sec: float
    timestamp: float


class TrialRunner:
    def __init__(
        self,
        config: EvalConfig,
        corpus: FactCorpus,
        probe: EngramProbe,
        judge: ClaudeJudge,
    ):
        self.config = config
        self.corpus = corpus
        self.probe = probe
        self.judge = judge
        self.records: list[TrialRecord] = []
        self._distractor_counter = 0
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def run_all(self, resume: bool = False):
        """Execute the full schedule sequentially.

        If resume=True, skip any trial whose output file already exists
        and reload its record from disk instead of re-running it.
        Engram memory state is NOT replayed for skipped trials — episodic
        and neural state persists in the project directory between runs,
        so the memory substrate remains consistent as long as the same
        project directory is used.
        """
        print(f"[runner] Starting eval — {len(self.config.schedule)} trials"
              + (" (resume mode)" if resume else ""))

        for trial_idx, entry in enumerate(self.config.schedule):
            label, subset_name, reps, n_distractors = entry

            # Check for existing result file
            trial_path = self._output_dir / f"trial_{trial_idx:02d}_{label}.json"
            if resume and trial_path.exists():
                print(f"\n[runner] Trial {trial_idx}: {label} — SKIPPING (already complete)")
                import json as _json
                raw = _json.loads(trial_path.read_text())
                import dataclasses
                known = {f.name for f in dataclasses.fields(TrialRecord)}
                record = TrialRecord(**{k: v for k, v in raw.items() if k in known})
                self.records.append(record)
                self._print_trial_summary(record)
                continue

            print(f"\n[runner] Trial {trial_idx}: {label}")
            record = await self._run_trial(
                trial_index=trial_idx,
                label=label,
                subset_name=subset_name,
                repetitions=reps,
                n_distractors=n_distractors,
            )
            self.records.append(record)
            self._save_trial(record)
            self._print_trial_summary(record)

        self._save_all()
        print(f"\n[runner] Complete. Results in {self._output_dir}/")

    async def _run_trial(
        self,
        trial_index: int,
        label: str,
        subset_name: str,
        repetitions: int,
        n_distractors: int,
    ) -> TrialRecord:
        t0 = time.perf_counter()
        injection_results = []
        target_facts = self.corpus.subset(subset_name)
        use_contradiction = (subset_name == "contradictions")

        # 1. Inject distractors (creates interference)
        if n_distractors > 0:
            print(f"  [inject] {n_distractors} distractor episodes")
            dist_tasks = [
                self.probe.inject_distractor(self._distractor_counter + i)
                for i in range(n_distractors)
            ]
            dist_results = await asyncio.gather(*dist_tasks)
            self._distractor_counter += n_distractors
            for r in dist_results:
                injection_results.append(asdict(r))

        # 2. Inject target facts (N repetitions)
        if target_facts and repetitions > 0:
            print(f"  [inject] {len(target_facts)} facts × {repetitions} reps"
                  f"{'  (contradictions)' if use_contradiction else ''}")
            for rep in range(repetitions):
                rep_tasks = [
                    self.probe.inject_fact(
                        fact=f,
                        use_contradiction=use_contradiction,
                        trial=trial_index,
                        repetition=rep,
                    )
                    for f in target_facts
                ]
                rep_results = await asyncio.gather(*rep_tasks)
                for r in rep_results:
                    injection_results.append(asdict(r))
                    # Update fact metadata
                    fact = self.corpus.by_id(r.fact_id)
                    fact.injected_count += 1
                    fact.last_injected_trial = trial_index

        # 3. Flush working memory before query if this is cold_query trial
        if label == "cold_query":
            print("  [probe] Flushing working memory (cold session simulation)")
            await self.probe.reset_working_memory()

        # 4. Retrieve all facts (all query types, max top_k)
        print(f"  [retrieve] Querying all {len(self.corpus)} facts × 3 query types")
        all_facts = self.corpus.facts
        max_k = max(self.config.retrieve_top_k)
        query_types = ["direct", "paraphrase", "decoy"]

        retrieval_tasks = [
            self.probe.retrieve(fact, qt, top_k=max_k)
            for fact in all_facts
            for qt in query_types
        ]
        retrievals = await asyncio.gather(*retrieval_tasks)

        # 5. Judge all retrievals
        print(f"  [judge] Sending {len(retrievals)} retrievals to Claude judge")
        facts_expanded = [
            fact
            for fact in all_facts
            for _ in query_types
        ]
        judgments = await self.judge.judge_batch(
            facts=facts_expanded,
            retrievals=retrievals,
            trial=trial_index,
        )

        elapsed = time.perf_counter() - t0
        return TrialRecord(
            trial_index=trial_index,
            label=label,
            subset_name=subset_name,
            repetitions=repetitions,
            distractors_injected=n_distractors,
            injection_results=injection_results,
            judgment_results=[
                {**j.as_dict(),
                 "retrieved_chunks": ret.retrieved_chunks,
                 "query": ret.query}
                for j, ret in zip(judgments, retrievals)
            ],
            elapsed_sec=elapsed,
            timestamp=time.time(),
        )

    def _save_trial(self, record: TrialRecord):
        path = self._output_dir / f"trial_{record.trial_index:02d}_{record.label}.json"
        path.write_text(json.dumps(asdict(record), indent=2))

    def _save_all(self):
        path = self._output_dir / "all_trials.json"
        path.write_text(json.dumps([asdict(r) for r in self.records], indent=2))

    def _print_trial_summary(self, record: TrialRecord):
        judgments = record.judgment_results
        if not judgments:
            print("  [summary] No judgments (inject-only trial)")
            return
        direct = [j for j in judgments if j["query_type"] == "direct"]
        paraphrase = [j for j in judgments if j["query_type"] == "paraphrase"]
        decoy = [j for j in judgments if j["query_type"] == "decoy"]

        def recall(subset):
            if not subset:
                return 0.0
            return sum(1 for j in subset if j["correct"]) / len(subset)

        def mean_relevance(subset):
            if not subset:
                return 0.0
            return sum(j["relevance"] for j in subset) / len(subset)

        contamination = sum(1 for j in judgments if j["contaminated"]) / max(len(judgments), 1)

        novelty_scores = [
            r.get("novelty_score")
            for r in record.injection_results
            if r.get("novelty_score") is not None
        ]
        novelty_str = (
            f"{sum(novelty_scores)/len(novelty_scores):.3f}"
            if novelty_scores else "N/A"
        )

        print(f"  recall(direct)={recall(direct):.2%}  "
              f"recall(paraphrase)={recall(paraphrase):.2%}  "
              f"decoy_resist={recall(decoy):.2%}")
        print(f"  mean_relevance={mean_relevance(judgments):.2f}/5  "
              f"contamination={contamination:.2%}  "
              f"mean_novelty={novelty_str}  "
              f"elapsed={record.elapsed_sec:.1f}s")
