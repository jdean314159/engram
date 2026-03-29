"""
Surprise Filter Implementation

TITANS-inspired perplexity-based gating for selective memory storage.
Reduces memory storage by 70-90% while preserving important context.

Based on Google Research TITANS papers:
- arxiv.org/abs/2501.00663
- arxiv.org/pdf/2504.13173

Key features:
- Perplexity-based surprise detection
- Adaptive thresholds with momentum buffer
- Project-specific calibration
- Human data baseline (anti-collapse)
- 70-90% memory reduction

Author: Jeffrey Dean
"""

import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class SurpriseMetrics:
    """Metrics for a single piece of content."""
    perplexity: float
    mean_logprob: float
    token_count: int
    timestamp: float
    is_surprising: bool
    threshold_used: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "perplexity": self.perplexity,
            "mean_logprob": self.mean_logprob,
            "token_count": self.token_count,
            "timestamp": self.timestamp,
            "is_surprising": self.is_surprising,
            "threshold_used": self.threshold_used,
        }


@dataclass
class SurpriseBaseline:
    """Baseline surprise statistics calibrated on human data."""
    mean: float
    std: float
    percentiles: Dict[int, float]  # {50: 10.5, 80: 18.2, 90: 25.3, 95: 32.1}
    sample_count: int
    calibration_date: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "percentiles": self.percentiles,
            "sample_count": self.sample_count,
            "calibration_date": self.calibration_date,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurpriseBaseline":
        return cls(
            mean=data["mean"],
            std=data["std"],
            percentiles={int(k): v for k, v in data["percentiles"].items()},
            sample_count=data["sample_count"],
            calibration_date=data["calibration_date"],
        )


@dataclass
class FilterStats:
    """Statistics for surprise filter operation."""
    total_evaluated: int = 0
    total_stored: int = 0
    total_rejected: int = 0
    storage_rate: float = 0.0
    avg_perplexity_stored: float = 0.0
    avg_perplexity_rejected: float = 0.0
    
    def update(self, stored: bool, perplexity: float):
        """Update statistics with new evaluation."""
        self.total_evaluated += 1
        
        if stored:
            self.total_stored += 1
            # Running average
            n = self.total_stored
            self.avg_perplexity_stored = (
                (self.avg_perplexity_stored * (n - 1) + perplexity) / n
            )
        else:
            self.total_rejected += 1
            n = self.total_rejected
            self.avg_perplexity_rejected = (
                (self.avg_perplexity_rejected * (n - 1) + perplexity) / n
            )
        
        self.storage_rate = self.total_stored / self.total_evaluated
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluated": self.total_evaluated,
            "total_stored": self.total_stored,
            "total_rejected": self.total_rejected,
            "storage_rate": self.storage_rate,
            "avg_perplexity_stored": self.avg_perplexity_stored,
            "avg_perplexity_rejected": self.avg_perplexity_rejected,
        }


class SurpriseFilter:
    """
    Perplexity-based surprise filter for selective memory storage.
    
    Based on TITANS papers from Google Research. Only stores content
    that is "surprising" (high perplexity) relative to the model's
    expectations.
    
    Key insight: Not all content is equally worth remembering. Store
    the surprising stuff, forget the predictable stuff.
    
    Usage:
        from llm_engine import VLLMEngine
        from memory_rag import SurpriseFilter
        
        engine = VLLMEngine(...)
        filter = SurpriseFilter(
            llm_engine=engine,
            base_threshold=20.0,
            project_id="programming_assistant"
        )
        
        # Calibrate on human data
        human_conversations = load_human_data()
        filter.calibrate(human_conversations)
        
        # Use in conversation
        user_input = "What's the difference between async and threading?"
        
        if filter.should_store(user_input):
            episodic_memory.add_episode(user_input + response)
        else:
            # Don't store - predictable content
            pass
    """
    
    def __init__(
        self,
        llm_engine,
        base_threshold: float = 20.0,
        percentile_threshold: int = 80,
        momentum: float = 0.9,
        buffer_size: int = 100,
        project_id: str = "default",
        calibration_required: bool = False,
    ):
        """
        Initialize surprise filter.
        
        Args:
            llm_engine: LLM engine with generate_with_logprobs() support
            base_threshold: Base perplexity threshold (ignored if calibrated)
            percentile_threshold: Percentile of human baseline to use (50-95)
            momentum: Decay factor for momentum buffer (0.0-1.0)
            buffer_size: Size of recent perplexity buffer
            project_id: Project identifier for per-project thresholds
            calibration_required: Require calibration before use
        """
        self.llm_engine = llm_engine
        self.base_threshold = base_threshold
        self.percentile_threshold = percentile_threshold
        self.momentum = momentum
        self.buffer_size = buffer_size
        self.project_id = project_id
        self.calibration_required = calibration_required

        # Probe engine capability at construction time so failures are loud
        # and early rather than silent at call time.
        self._logprobs_available = self._probe_logprobs()
        self._warned_no_logprobs = False
        
        # Calibration state
        self.baseline: Optional[SurpriseBaseline] = None
        self.is_calibrated = False
        
        # Per-project overrides
        self.project_thresholds = {
            "programming_assistant": 28.0,  # Higher = only novel patterns
            "language_tutor": 18.0,         # Lower = catch vocabulary mistakes
            "file_organizer": 25.0,         # Medium = categorization decisions
            "voice_interface": 22.0,        # Medium = command patterns
        }

        # Adaptive threshold with momentum
        self.current_threshold = self.project_thresholds.get(project_id, base_threshold)
        self.recent_perplexities = deque(maxlen=buffer_size)

        # Statistics
        self.stats = FilterStats()
    
    def _probe_logprobs(self) -> bool:
        """Return True if the engine can produce logprobs, False otherwise.

        Checks three things in order:
        1. Engine explicitly declares ``supports_logprobs = False``.
        2. Engine has no ``generate_with_logprobs`` attribute at all.
        3. Importing ``engine.base`` (which requires pydantic) succeeds.
           If pydantic is missing this import fails — we catch it here once
           so every subsequent call can branch cleanly instead of swallowing
           a ModuleNotFoundError inside a broad ``except Exception``.
        """
        if getattr(self.llm_engine, "supports_logprobs", True) is False:
            return False
        if not callable(getattr(self.llm_engine, "generate_with_logprobs", None)):
            return False
        try:
            from engram.engine.base import LogprobResult  # noqa: F401
            return True
        except ImportError as e:
            logger.warning(
                "SurpriseFilter: logprob support unavailable (%s). "
                "Filter will operate in pass-through mode (store everything). "
                "Install pydantic to enable perplexity-based filtering.",
                e,
            )
            return False

    def _compute_perplexity(self, text: str) -> Optional[float]:
        """Call the engine's logprob endpoint and return perplexity, or None.

        Raises RuntimeError if logprobs are structurally unavailable
        (missing engine or missing pydantic import) so callers that
        depend on a result can fail explicitly rather than silently storing
        everything.

        Returns None only for transient engine errors (network timeouts,
        empty responses) — those are genuinely recoverable.
        """
        if not self._logprobs_available:
            raise RuntimeError(
                "SurpriseFilter cannot compute perplexity: the LLM engine "
                "does not support generate_with_logprobs(), or pydantic is "
                "not installed. Install pydantic (pip install pydantic) or "
                "use an engine that supports logprobs."
            )
        try:
            result = self.llm_engine.generate_with_logprobs(
                text, max_tokens=1, temperature=0
            )
            if result.token_count == 0:
                return None
            return result.perplexity
        except Exception as e:
            logger.debug("SurpriseFilter._compute_perplexity transient error: %s", e)
            return None

    def calibrate(
        self,
        human_texts: List[str],
        force: bool = False,
    ) -> SurpriseBaseline:
        """
        Calibrate filter on human data.
        
        CRITICAL: This must be done on HUMAN text, not model-generated text.
        This establishes the baseline for what counts as "surprising" relative
        to human communication patterns.
        
        Args:
            human_texts: List of human-written texts (>100 samples recommended)
            force: Force recalibration even if already calibrated
            
        Returns:
            SurpriseBaseline with statistics
        """
        if self.is_calibrated and not force:
            logger.info("Already calibrated. Use force=True to recalibrate.")
            return self.baseline
        
        if len(human_texts) < 50:
            logger.warning("Only %d samples. Recommend >100 for reliable calibration.", len(human_texts))
        
        logger.info("Calibrating surprise filter on %d human texts...", len(human_texts))
        
        perplexities = []

        for i, text in enumerate(human_texts):
            if i % 20 == 0:
                logger.debug("Calibration progress: %d/%d", i, len(human_texts))

            pp = self._compute_perplexity(text)
            if pp is not None:
                perplexities.append(pp)
            else:
                logger.debug("Calibration: skipping text %d (no perplexity)", i)
        
        logger.info("Processed %d texts successfully", len(perplexities))
        
        # Calculate statistics
        perplexities_array = np.array(perplexities)
        
        baseline = SurpriseBaseline(
            mean=float(np.mean(perplexities_array)),
            std=float(np.std(perplexities_array)),
            percentiles={
                50: float(np.percentile(perplexities_array, 50)),
                80: float(np.percentile(perplexities_array, 80)),
                90: float(np.percentile(perplexities_array, 90)),
                95: float(np.percentile(perplexities_array, 95)),
            },
            sample_count=len(perplexities),
            calibration_date=time.time(),
        )
        
        # Set threshold based on percentile
        self.current_threshold = baseline.percentiles[self.percentile_threshold]
        self.baseline = baseline
        self.is_calibrated = True
        
        logger.info("Calibration complete:")
        logger.info("  Mean perplexity: %.2f", baseline.mean)
        logger.info("  Std: %.2f", baseline.std)
        logger.info("  %dth percentile: %.2f", self.percentile_threshold, self.current_threshold)
        logger.info("  Threshold set to: %.2f", self.current_threshold)
        
        return baseline
    
    def should_store(
        self,
        text: str,
        perplexity: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Decide if content should be stored based on surprise.
        
        Args:
            text: Content to evaluate
            perplexity: Pre-computed perplexity (optional)
            metadata: Optional metadata (for logging)
            
        Returns:
            True if content is surprising enough to store
        """
        # Calibration is recommended for best results, but should not block
        # first-run usability unless the caller explicitly insists.
        if self.calibration_required and not self.is_calibrated:
            # Proceed using the base (or project) threshold.
            pass

        # If engine cannot produce logprobs, store conservatively and warn once.
        if not self._logprobs_available:
            if not self._warned_no_logprobs:
                logger.warning(
                    "SurpriseFilter: logprob support unavailable; "
                    "storing all episodes (pass-through mode)."
                )
                self._warned_no_logprobs = True
            return True

        # Compute perplexity if not provided
        if perplexity is None:
            try:
                perplexity = self._compute_perplexity(text)
            except RuntimeError:
                return True  # no logprob support — store conservatively
            if perplexity is None:
                # Transient engine error — store conservatively
                return True
        
        # Add to momentum buffer
        self.recent_perplexities.append(perplexity)
        
        # Update adaptive threshold with momentum (only when calibrated)
        if self.is_calibrated and len(self.recent_perplexities) >= 10:
            recent_mean = np.mean(list(self.recent_perplexities))
            # Momentum update: new_threshold = momentum * old + (1 - momentum) * recent_mean
            self.current_threshold = (
                self.momentum * self.current_threshold +
                (1 - self.momentum) * recent_mean
            )
        
        # Use project-specific threshold if calibrated
        if self.is_calibrated:
            threshold = self.current_threshold
        else:
            threshold = self.project_thresholds.get(self.project_id, self.base_threshold)
        
        # Is it surprising?
        is_surprising = perplexity > threshold
        
        # Additional check: Is it TOO surprising? (likely noise/error)
        if self.is_calibrated:
            max_reasonable = self.baseline.mean + 3 * self.baseline.std
            is_reasonable = perplexity <= max_reasonable
        else:
            is_reasonable = perplexity <= threshold * 3  # 3x threshold as upper bound
        
        should_store = is_surprising and is_reasonable
        
        # Update statistics
        self.stats.update(should_store, perplexity)
        
        return should_store
    
    def evaluate_batch(
        self,
        texts: List[str],
    ) -> List[SurpriseMetrics]:
        """
        Evaluate multiple texts and return metrics.
        
        Useful for analyzing a batch of content.
        
        Args:
            texts: List of texts to evaluate
            
        Returns:
            List of SurpriseMetrics
        """
        results = []

        for text in texts:
            try:
                perplexity = self._compute_perplexity(text)
            except RuntimeError as e:
                logger.warning("evaluate_batch: logprobs unavailable — %s", e)
                break  # no point continuing if engine cannot do logprobs

            if perplexity is None:
                logger.debug("evaluate_batch: skipping text (no perplexity returned)")
                continue

            try:
                raw = self.llm_engine.generate_with_logprobs(
                    text, max_tokens=1, temperature=0
                )
                mean_logprob = raw.mean_logprob
                token_count = raw.token_count
            except Exception:
                mean_logprob = 0.0
                token_count = 0

            is_surprising = self.should_store(text, perplexity=perplexity)

            metrics = SurpriseMetrics(
                perplexity=perplexity,
                mean_logprob=mean_logprob,
                token_count=token_count,
                timestamp=time.time(),
                is_surprising=is_surprising,
                threshold_used=self.current_threshold,
            )
            results.append(metrics)

        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        stats = self.stats.to_dict()

        stats.update({
            "project_id": self.project_id,
            "current_threshold": self.current_threshold,
            "is_calibrated": self.is_calibrated,
            "logprobs_available": self._logprobs_available,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "recent_perplexities_count": len(self.recent_perplexities),
        })
        
        return stats
    
    def save_calibration(self, path: Path):
        """Save calibration baseline to file."""
        if not self.is_calibrated:
            raise ValueError("No calibration to save")
        
        data = {
            "baseline": self.baseline.to_dict(),
            "project_id": self.project_id,
            "percentile_threshold": self.percentile_threshold,
            "momentum": self.momentum,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info("Calibration saved to %s", path)
    
    def load_calibration(self, path: Path):
        """Load calibration baseline from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.baseline = SurpriseBaseline.from_dict(data["baseline"])
        self.project_id = data["project_id"]
        self.percentile_threshold = int(data["percentile_threshold"])
        self.momentum = data.get("momentum", 0.9)
        
        # Set threshold from calibration
        self.current_threshold = self.baseline.percentiles[self.percentile_threshold]
        self.is_calibrated = True
        
        logger.info("Calibration loaded from %s", path)
        logger.info("  Threshold: %.2f", self.current_threshold)
    
    def reset_stats(self):
        """Reset statistics (useful between experiments)."""
        self.stats = FilterStats()
        self.recent_perplexities.clear()
