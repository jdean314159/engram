"""Validation tests for P1 Bug #3 — SurpriseFilter explicit failure modes.

Tests verify:
1. Construction-time probe: _logprobs_available is set correctly.
2. Missing pydantic → warned at construction, not silently mid-loop.
3. Engine with supports_logprobs=False → warned at construction.
4. Engine with no generate_with_logprobs method → warned at construction.
5. should_store() → True (conservative) when logprobs unavailable, not silent.
6. evaluate_batch() → empty list (graceful) when logprobs unavailable.
7. calibrate() skips texts that return None, completes without crash.
8. _compute_perplexity() raises RuntimeError explicitly when unavailable.
9. get_stats() surfaces logprobs_available key.
10. With a working engine, normal filter behaviour is preserved.
"""

from __future__ import annotations

from typing import List, Optional

from .runner import test_group, require
from .mocks import TempDir


# ── Helpers ────────────────────────────────────────────────────────────────────

class _NoLogprobsEngine:
    """Engine that explicitly declares it cannot produce logprobs."""
    supports_logprobs = False

    def generate_with_logprobs(self, text, **kw):
        raise NotImplementedError("No logprobs")


class _NoMethodEngine:
    """Engine that has no generate_with_logprobs attribute at all."""
    pass


class _BrokenImportEngine:
    """Engine whose generate_with_logprobs call triggers an ImportError
    (simulates pydantic absent — the _probe_logprobs check catches this)."""

    def generate_with_logprobs(self, text, **kw):
        # Simulate what happens when engine.base can't be imported
        raise ImportError("No module named 'pydantic'")


class _WorkingEngine:
    """Engine that returns valid logprob results."""

    def __init__(self, perplexity: float = 12.0):
        self._perplexity = perplexity

    def generate_with_logprobs(self, text, **kw):
        # Return a duck-type result (no pydantic needed)
        class _Result:
            token_count = 5
            perplexity = 0.0
            mean_logprob = -2.0
        r = _Result()
        r.perplexity = self._perplexity
        return r


def _make_sf(engine, **kw):
    from engram.filters.surprise_filter import SurpriseFilter
    return SurpriseFilter(engine, **kw)


# ── Construction-time probe ────────────────────────────────────────────────────

@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_probe_no_logprobs_flag():
    """Engine with supports_logprobs=False → probe returns False immediately."""
    sf = _make_sf(_NoLogprobsEngine())
    assert sf._logprobs_available is False


@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_probe_no_method():
    """Engine with no generate_with_logprobs attr → probe returns False."""
    sf = _make_sf(_NoMethodEngine())
    assert sf._logprobs_available is False


@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_probe_missing_pydantic():
    """_probe_logprobs correctly classifies engine capability.

    _BrokenImportEngine.generate_with_logprobs raises ImportError at call time.
    _probe_logprobs only checks (a) supports_logprobs flag, (b) method exists,
    (c) engram.engine.base import succeeds — it does NOT call the engine.

    When pydantic IS installed:
      - engram.engine.base import succeeds → _logprobs_available=True
      - engine failure happens at runtime (compute_perplexity returns None)

    When pydantic is NOT installed:
      - engram.engine.base import fails → _logprobs_available=False
    """
    from .runner import DEPS
    sf = _make_sf(_BrokenImportEngine())

    if DEPS.get("pydantic"):
        # Import check passes → probe reports available
        assert sf._logprobs_available is True
        # Runtime call fails silently → returns None (transient error path)
        result = sf._compute_perplexity("test text")
        assert result is None, (
            f"Expected None (engine raises ImportError at call time), got {result!r}"
        )
    else:
        # Import check fails → probe warns and reports unavailable
        assert sf._logprobs_available is False


@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_probe_working_engine():
    """Working engine → probe returns True (requires pydantic for engine.base import)."""
    require("pydantic")
    sf = _make_sf(_WorkingEngine())
    assert sf._logprobs_available is True


# ── get_stats exposes logprobs_available ───────────────────────────────────────

@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_get_stats_has_logprobs_available_key():
    sf = _make_sf(_NoLogprobsEngine())
    stats = sf.get_stats()
    assert "logprobs_available" in stats
    assert stats["logprobs_available"] is False


@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_get_stats_logprobs_true_for_working_engine():
    require("pydantic")
    sf = _make_sf(_WorkingEngine())
    stats = sf.get_stats()
    assert stats["logprobs_available"] is True


# ── _compute_perplexity raises explicitly ──────────────────────────────────────

@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_compute_perplexity_raises_runtime_error_when_unavailable():
    """_compute_perplexity raises RuntimeError — not swallows silently."""
    sf = _make_sf(_NoLogprobsEngine())
    try:
        sf._compute_perplexity("some text")
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "pydantic" in str(e).lower() or "logprobs" in str(e).lower()


@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_compute_perplexity_returns_value_when_available():
    """_compute_perplexity returns a float from a working engine."""
    require("pydantic")
    sf = _make_sf(_WorkingEngine(perplexity=15.0))
    pp = sf._compute_perplexity("some text")
    assert isinstance(pp, float)
    assert pp == 15.0


@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_compute_perplexity_returns_none_on_transient_error():
    """Transient engine error (e.g. timeout) → None, not exception."""
    class _TransientEngine:
        def generate_with_logprobs(self, text, **kw):
            raise ConnectionError("timeout")

    sf = _make_sf(_TransientEngine())
    # _probe_logprobs passes (method exists, import succeeds — no pydantic check needed
    # since _WorkingEngine path). Force _logprobs_available=True to test the transient path.
    sf._logprobs_available = True
    result = sf._compute_perplexity("some text")
    assert result is None


# ── should_store conservative behaviour ───────────────────────────────────────

@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_should_store_true_when_no_logprobs():
    """should_store returns True (store conservatively) when unavailable."""
    sf = _make_sf(_NoLogprobsEngine())
    result = sf.should_store("Store this conservatively", perplexity=None)
    assert result is True


@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_should_store_warns_once_not_every_call():
    """Warning about missing logprobs is emitted only once, not per call."""
    with _capture_warnings() as warnings:
        sf = _make_sf(_NoLogprobsEngine())
        for _ in range(5):
            sf.should_store("text", perplexity=None)
    # Construction warning + at most one should_store warning
    logprob_warnings = [w for w in warnings if "pass-through" in w.lower()
                        or "logprob" in w.lower()]
    assert len(logprob_warnings) <= 2


@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_should_store_uses_precomputed_perplexity():
    """should_store with explicit perplexity skips engine call entirely."""
    call_count = {"n": 0}

    class _CountingEngine(_WorkingEngine):
        def generate_with_logprobs(self, text, **kw):
            call_count["n"] += 1
            return super().generate_with_logprobs(text, **kw)

    sf = _make_sf(_CountingEngine(perplexity=5.0), base_threshold=50.0)
    # Pass perplexity directly — engine should not be called
    sf.should_store("text", perplexity=5.0)
    assert call_count["n"] == 0


@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_should_store_filters_below_threshold():
    """With working engine and low perplexity → False (not stored)."""
    require("pydantic")
    sf = _make_sf(_WorkingEngine(perplexity=2.0), base_threshold=50.0,
                  calibration_required=False)
    result = sf.should_store("mundane text", perplexity=2.0)
    assert result is False


@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_should_store_passes_above_threshold():
    """Perplexity above threshold but below noise cap (threshold*3) → True.
    
    The filter guards against noise by rejecting perplexity > threshold*3.
    threshold=5.0 → valid range for storage: (5.0, 15.0).
    """
    sf = _make_sf(_WorkingEngine(perplexity=10.0), base_threshold=5.0,
                  calibration_required=False)
    # perplexity=10.0: above 5.0 threshold AND below 15.0 noise cap
    result = sf.should_store("moderately surprising text", perplexity=10.0)
    assert result is True, (
        f"Expected True (10.0 in range (5.0, 15.0)), got {result}. "
        f"threshold={sf.current_threshold}"
    )


@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_should_store_filtered_above_noise_cap():
    """Perplexity above threshold*3 (noise cap) is rejected as noise → False.

    The filter has two rejection gates:
      below threshold           → False (not surprising enough)
      above threshold * 3      → False (too surprising, likely noise/error)
      in (threshold, threshold*3) → True  (genuinely surprising, store it)

    This test verifies the upper noise cap: perplexity=100.0 with
    threshold=5.0 gives cap=15.0, so 100.0 must return False.

    Requires pydantic: without it the filter runs in pass-through mode
    (returns True for everything) and the noise cap is never reached.
    """
    require("pydantic")
    sf = _make_sf(_WorkingEngine(perplexity=100.0), base_threshold=5.0,
                  calibration_required=False)
    result = sf.should_store("potentially noisy outlier text", perplexity=100.0)
    assert result is False, (
        f"Expected False (100.0 > noise cap {sf.current_threshold * 3:.1f}), "
        f"got {result}. threshold={sf.current_threshold}"
    )


# ── evaluate_batch graceful when unavailable ───────────────────────────────────

@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_evaluate_batch_empty_when_no_logprobs():
    """evaluate_batch returns [] without raising when logprobs unavailable."""
    sf = _make_sf(_NoLogprobsEngine())
    result = sf.evaluate_batch(["text one", "text two", "text three"])
    assert result == []


@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_evaluate_batch_returns_metrics_when_working():
    """evaluate_batch returns one SurpriseMetrics per input with working engine."""
    require("pydantic")
    from engram.filters.surprise_filter import SurpriseMetrics
    # Use perplexity=10.0 (above threshold=5.0 but below 5.0*3=15.0)
    sf = _make_sf(_WorkingEngine(perplexity=10.0), base_threshold=5.0,
                  calibration_required=False)
    results = sf.evaluate_batch(["text one", "text two"])
    assert len(results) == 2
    for m in results:
        assert isinstance(m, SurpriseMetrics)
        assert m.perplexity == 10.0
        assert isinstance(m.is_surprising, bool)
        assert isinstance(m.threshold_used, float)
        assert m.is_surprising is True  # 10.0 in (5.0, 15.0)


# ── calibrate graceful when texts produce None ─────────────────────────────────

@test_group("P1-3: SurpriseFilter Explicit Failures")
def test_calibrate_skips_none_results():
    """calibrate skips texts where perplexity returns None (transient errors)."""
    require("pydantic")
    call_count = {"n": 0}

    class _SometimesNoneEngine(_WorkingEngine):
        def generate_with_logprobs(self, text, **kw):
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                # Return a zero-token result so _compute_perplexity returns None
                class _Empty:
                    token_count = 0
                    perplexity = 0.0
                    mean_logprob = 0.0
                return _Empty()
            return super().generate_with_logprobs(text, **kw)

    sf = _make_sf(_SometimesNoneEngine(perplexity=12.0))
    texts = [f"Text {i}" for i in range(10)]
    baseline = sf.calibrate(texts)
    # Should have processed ~half the texts without crashing
    assert baseline.sample_count > 0
    assert sf.is_calibrated


# ── Utility ────────────────────────────────────────────────────────────────────

import contextlib
import logging

@contextlib.contextmanager
def _capture_warnings():
    """Capture WARNING-level log messages emitted during the block."""
    captured = []

    class _Handler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                captured.append(record.getMessage())

    handler = _Handler()
    logger = logging.getLogger("engram.filters.surprise_filter")
    logger.addHandler(handler)
    try:
        yield captured
    finally:
        logger.removeHandler(handler)
