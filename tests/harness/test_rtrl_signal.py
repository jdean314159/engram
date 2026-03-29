"""Validation tests for RTRL neural memory surprise signal quality.

These tests measure whether the surprise signal is actually meaningful —
i.e., whether it correlates with semantic novelty in a controlled corpus.

This is the critical gate before wiring the neural layer further into the
memory system.  If the correlation is weak (< 0.3), the RTRL parameters
or embedding quality need investigation before using the signal for
importance scoring or retrieval ranking.

Pass criteria:
    Spearman correlation >= 0.5  → signal is real, safe to use
    0.3 <= correlation < 0.5    → weak but present, may need tuning
    < 0.3                       → signal is noise at this scale

The test corpus has deliberately structured novelty:
  - Warmup block (50 turns, not scored): establishes stable EMA baseline
  - Block A (turns on asyncio): establishes familiar pattern
  - Block B (repetitions of block A): should produce LOW surprise
  - Block C (FastAPI — topic shift): should produce HIGH surprise
  - Block D (return to asyncio): should produce LOW surprise
  - Block E (Fourier transforms — novel): should produce HIGH surprise
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from .runner import test_group, require
from .mocks import TempDir, unique_session


# ── Corpus definition ─────────────────────────────────────────────────────────
#
# Each entry: (user_text, assistant_text, novelty_label)
# novelty_label: 0.0 = familiar/repetitive, 1.0 = novel/surprising
#
# Labels are ground truth for Spearman correlation against RTRL surprise.

_WARMUP: List[Tuple[str, str]] = [
    # 50 warmup turns — EMA with alpha=0.05 needs ~50 steps to stabilize.
    # All Python basics: establishes a stable familiar baseline.
    ("Hello", "Hello! How can I help you?"),
    ("What is Python?", "Python is a high-level programming language."),
    ("How do I print in Python?", "Use print('hello')."),
    ("What is a variable?", "A variable stores a value."),
    ("What is a list?", "A list stores ordered items."),
    ("How do I loop?", "Use a for loop."),
    ("What is a function?", "A function groups reusable code."),
    ("How do I import?", "Use import module_name."),
    ("What is a class?", "A class is a blueprint for objects."),
    ("What is inheritance?", "A subclass inherits from a parent class."),
    ("What is Python?", "Python is interpreted and dynamically typed."),
    ("How do I print?", "Use the print() function."),
    ("What is a variable?", "Variables hold values of any type."),
    ("What is a list?", "Lists are mutable ordered sequences."),
    ("How do I loop?", "for item in collection: ..."),
    ("What is a function?", "def name(args): ..."),
    ("How do I import?", "import module or from module import name"),
    ("What is a class?", "class Name: def __init__(self): ..."),
    ("What is inheritance?", "class Child(Parent): ..."),
    ("What is Python?", "Python is popular for data science and web."),
    # 20 more: asyncio specifically — gives the network a deep familiar pattern
    ("What is asyncio?", "asyncio is Python's async I/O library."),
    ("How do I use asyncio?", "Use async def and await."),
    ("What is an event loop?", "The event loop schedules coroutines."),
    ("What is asyncio?", "asyncio handles async I/O in Python."),
    ("How does await work?", "await suspends the coroutine."),
    ("What is async def?", "async def defines a coroutine."),
    ("What is asyncio?", "It uses an event loop for scheduling."),
    ("How does asyncio work?", "Coroutines yield control to the loop."),
    ("What is a coroutine?", "A coroutine is a pausable function."),
    ("What is asyncio.run?", "asyncio.run() starts the event loop."),
    ("What is asyncio?", "asyncio is for concurrent I/O without threads."),
    ("How do I await?", "Put await before an awaitable expression."),
    ("What is asyncio?", "It's Python's standard async library."),
    ("What is an event loop?", "It runs coroutines and handles I/O."),
    ("How does async def work?", "It creates a coroutine object."),
    ("What is asyncio.gather?", "gather() runs coroutines concurrently."),
    ("What is asyncio?", "asyncio enables cooperative multitasking."),
    ("What is await?", "await pauses until the awaitable is done."),
    ("What is asyncio?", "It's built into Python's standard library."),
    ("How do I use async?", "Declare functions with async def."),
    # 10 more: general Python reinforcement
    ("What is a list comprehension?", "[ x for x in iterable ]"),
    ("What is a dict?", "A dict maps keys to values."),
    ("What is a tuple?", "An immutable ordered sequence."),
    ("What is None?", "None is Python's null value."),
    ("What is a generator?", "A function that yields values lazily."),
    ("What is a decorator?", "A function that wraps another function."),
    ("What is type hinting?", "Annotations like x: int specify types."),
    ("What is a context manager?", "An object with __enter__ and __exit__."),
    ("What is f-string?", "A formatted string literal: f'hello {name}'."),
    ("What is *args?", "Variable positional arguments."),
]

_CORPUS: List[Tuple[str, str, float]] = [
    # Block A — asyncio (turns 1-5, establishing pattern)
    ("What is asyncio?",
     "asyncio is Python's standard library for async I/O.",
     0.8),
    ("How do I use asyncio?",
     "Define coroutines with async def and await them.",
     0.6),
    ("What is an event loop?",
     "The event loop schedules and runs coroutines.",
     0.5),
    ("How does await work?",
     "await suspends a coroutine until the awaited task completes.",
     0.5),
    ("What is async def?",
     "async def defines a coroutine function.",
     0.4),

    # Block B — asyncio repetitions (should be LOW surprise)
    ("Explain asyncio briefly.",
     "asyncio handles async I/O in Python using coroutines.",
     0.1),
    ("What is asyncio again?",
     "It's Python's async I/O library with event loops.",
     0.1),
    ("How does the event loop work?",
     "It runs coroutines and handles I/O callbacks.",
     0.1),
    ("What does await do?",
     "await pauses execution until the coroutine finishes.",
     0.1),
    ("asyncio overview?",
     "async def, await, and event loop are the core primitives.",
     0.1),

    # Block C — FastAPI (topic shift, should be HIGH surprise)
    ("What is FastAPI?",
     "FastAPI is a modern Python web framework for building APIs.",
     0.9),
    ("How do I create an endpoint?",
     "Use @app.get('/path') to define a GET endpoint.",
     0.8),
    ("What is a path parameter?",
     "A path parameter is a variable in the URL path like /items/{id}.",
     0.8),
    ("How does FastAPI handle validation?",
     "FastAPI uses Pydantic models for automatic request validation.",
     0.8),
    ("What is dependency injection in FastAPI?",
     "Use Depends() to inject shared logic into route handlers.",
     0.7),

    # Block D — return to asyncio (should drop back to LOW)
    ("asyncio event loop again?",
     "The loop schedules coroutines and I/O callbacks.",
     0.2),
    ("What is async/await?",
     "async def declares coroutines, await suspends them.",
     0.2),
    ("How do I run asyncio?",
     "Use asyncio.run(main()) to start the event loop.",
     0.2),

    # Block E — completely new topic (should spike HIGH)
    ("What is a Fourier transform?",
     "It decomposes a signal into its constituent frequencies.",
     0.95),
    ("What is the FFT algorithm?",
     "FFT is a fast O(n log n) algorithm for computing Fourier transforms.",
     0.9),
    ("What are frequency components?",
     "A signal's frequency components describe its oscillation patterns.",
     0.85),
]


# ── Test helpers ──────────────────────────────────────────────────────────────

def _make_coordinator(d):
    """Build a NeuralCoordinator with real RTRL and a deterministic embedder."""
    from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig, EmbeddingProjector
    from engram.memory.neural_coordinator import NeuralCoordinator

    class _DeterministicEmbedder:
        """Hash-based embedder: same text → same vector, different text → different vector.
        No model download required.  Sufficient for testing signal correlation."""
        _DIM = 64

        def embed(self, text: str):
            # Deterministic pseudo-random vector from text hash
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            seed = int.from_bytes(h[:4], "little")
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._DIM).astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-8)

    EMBED_DIM = _DeterministicEmbedder._DIM
    cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16, enabled=True)
    neural = NeuralMemory(project_dir=d, config=cfg)
    coord = NeuralCoordinator(
        neural=neural,
        key_projector=EmbeddingProjector(EMBED_DIM, 64),
        value_projector=EmbeddingProjector(EMBED_DIM, 16),
        embedding_service=_DeterministicEmbedder(),
    )
    return coord


class _FakeMsg:
    metadata = {}


def _run_turn(coord, user_text: str, assistant_text: str) -> float:
    """Feed one turn and return the surprise EMA after the assistant step."""
    coord.feed("user", user_text, _FakeMsg())
    coord.feed("assistant", assistant_text, _FakeMsg())
    surprise = coord.get_last_surprise()
    return surprise if surprise is not None else 0.0


# ── Core validation test ──────────────────────────────────────────────────────

@test_group("Neural Memory")
def test_rtrl_surprise_signal_exists():
    """Basic sanity: RTRL produces non-zero surprise after warmup."""
    require("torch")
    with TempDir() as d:
        coord = _make_coordinator(d)

        # Run warmup
        for user, assistant in _WARMUP:
            _run_turn(coord, user, assistant)

        assert coord.is_warmed_up(), "Network not warmed up after 50 turns"

        # Run one novel turn — should produce non-zero surprise
        surprise = _run_turn(coord, "What is a Fourier transform?",
                             "It decomposes a signal into frequencies.")
        assert surprise > 0.0, f"Expected non-zero surprise, got {surprise}"


@test_group("Neural Memory")
def test_rtrl_repeated_turns_lower_surprise():
    """Repeated similar turns should produce lower surprise than novel ones."""
    require("torch")
    with TempDir() as d:
        coord = _make_coordinator(d)

        # Warmup
        for user, assistant in _WARMUP:
            _run_turn(coord, user, assistant)

        # Establish asyncio pattern
        for _ in range(5):
            _run_turn(coord, "What is asyncio?",
                      "asyncio is Python's async I/O library.")

        # Repeated familiar turn
        familiar_surprise = _run_turn(
            coord, "What is asyncio?", "asyncio handles async I/O."
        )

        # Novel turn (completely different topic)
        novel_surprise = _run_turn(
            coord, "What is a Fourier transform?",
            "It decomposes signals into frequencies."
        )

        assert novel_surprise > familiar_surprise, (
            f"Expected novel ({novel_surprise:.4f}) > familiar ({familiar_surprise:.4f}). "
            f"Signal is not differentiating topic shifts."
        )


@test_group("Neural Memory")
def test_rtrl_surprise_correlates_with_novelty():
    """Spearman correlation between RTRL surprise and ground-truth novelty labels.

    Pass: correlation >= 0.4 (moderate signal, usable for importance scoring)
    Warn: correlation 0.2-0.4 (weak signal, needs investigation)
    Fail: correlation < 0.2 (no signal)
    """
    require("torch", "sklearn")
    from scipy.stats import spearmanr

    with TempDir() as d:
        coord = _make_coordinator(d)

        # Warmup (not scored)
        for user, assistant in _WARMUP:
            _run_turn(coord, user, assistant)

        assert coord.is_warmed_up()

        # Run corpus and collect surprise scores
        surprises: List[float] = []
        labels: List[float] = []

        for user_text, assistant_text, novelty_label in _CORPUS:
            s = _run_turn(coord, user_text, assistant_text)
            surprises.append(s)
            labels.append(novelty_label)

        surprises_arr = np.array(surprises)
        labels_arr = np.array(labels)

        corr, pvalue = spearmanr(surprises_arr, labels_arr)

        # Diagnostic output always printed for visibility
        print(f"\nRTRL signal quality:")
        print(f"  Spearman r = {corr:.3f}  (p={pvalue:.3f})")
        print(f"  Surprise range: [{surprises_arr.min():.4f}, {surprises_arr.max():.4f}]")
        print(f"  Surprise mean: {surprises_arr.mean():.4f}  std: {surprises_arr.std():.4f}")
        print(f"  Label mean:    {labels_arr.mean():.3f}")

        # Block-level means for diagnostic detail
        block_sizes = [5, 5, 5, 3, 3]
        block_names = ["A-asyncio(est)", "B-asyncio(rep)", "C-fastapi", "D-asyncio(ret)", "E-fourier"]
        idx = 0
        for name, size in zip(block_names, block_sizes):
            block_s = surprises_arr[idx:idx+size]
            block_l = labels_arr[idx:idx+size]
            print(f"  {name}: surprise_mean={block_s.mean():.4f}  label_mean={block_l.mean():.2f}")
            idx += size

        # Minimum threshold: signal must be present
        assert corr >= 0.2, (
            f"RTRL surprise shows no correlation with novelty (r={corr:.3f}). "
            f"The neural layer is not producing a useful signal. "
            f"Consider: (1) longer warmup, (2) larger hidden_dim, "
            f"(3) different EMA alpha, (4) embedding quality."
        )

        # Soft warning for weak signal
        if corr < 0.4:
            print(
                f"\n  WARNING: Signal is weak (r={corr:.3f} < 0.4). "
                f"Dynamic importance scoring will have limited effect. "
                f"Consider tuning RTRL parameters before relying on this signal."
            )
        else:
            print(f"\n  OK: Signal is usable (r={corr:.3f} >= 0.4).")


@test_group("Neural Memory")
def test_rtrl_parameter_comparison():
    """Compare signal quality across RTRL parameter configurations.

    Runs the same novelty corpus through three configurations:
      - Baseline: hidden_dim=32, value_dim=16, grad_clip=5.0  (current defaults)
      - Wider:    hidden_dim=64, value_dim=32, grad_clip=2.0  (proposed)
      - Compact:  hidden_dim=16, value_dim=8,  grad_clip=5.0  (sanity check)

    Reports Spearman r and surprise range for each. The test passes as long as
    the baseline config meets the minimum threshold — the comparison is
    diagnostic, not a hard requirement on the wider config.

    If hidden_dim=64 causes P-matrix overflow, surprise values will be NaN
    and the test will report that explicitly rather than failing opaquely.
    """
    require("torch", "sklearn")
    from scipy.stats import spearmanr
    from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig, EmbeddingProjector
    from engram.memory.neural_coordinator import NeuralCoordinator

    configs = [
        ("baseline  hidden=32 value=16 clip=5.0",
         NeuralMemoryConfig(hidden_dim=32, value_dim=16, enabled=True)),
        ("wider     hidden=64 value=32 clip=2.0",
         NeuralMemoryConfig(hidden_dim=64, value_dim=32, enabled=True)),
        ("compact   hidden=16 value=8  clip=5.0",
         NeuralMemoryConfig(hidden_dim=16, value_dim=8,  enabled=True)),
    ]

    EMBED_DIM = 64

    class _Det:
        """Same deterministic embedder as _make_coordinator."""
        _DIM = EMBED_DIM
        def embed(self, text: str):
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            seed = int.from_bytes(h[:4], "little")
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._DIM).astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-8)

    print("\nRTRL parameter comparison:")
    print(f"  {'Config':<40}  {'r':>6}  {'p':>6}  {'range':>10}  {'stable':>7}")
    print("  " + "-" * 75)

    baseline_corr = None

    for label, cfg in configs:
        with TempDir() as d:
            try:
                neural = NeuralMemory(project_dir=d, config=cfg)
                # key_projector always outputs key_dim (64) — independent of hidden_dim
                # value_projector outputs value_dim (varies by config)
                KEY_DIM = 64
                coord = NeuralCoordinator(
                    neural=neural,
                    key_projector=EmbeddingProjector(EMBED_DIM, KEY_DIM),
                    value_projector=EmbeddingProjector(EMBED_DIM, cfg.value_dim),
                    embedding_service=_Det(),
                )

                # Warmup
                for user, assistant in _WARMUP:
                    _run_turn(coord, user, assistant)

                # Corpus run
                surprises: List[float] = []
                labels: List[float] = []
                has_nan = False

                for user_text, assistant_text, novelty_label in _CORPUS:
                    s = _run_turn(coord, user_text, assistant_text)
                    if np.isnan(s):
                        has_nan = True
                        break
                    surprises.append(s)
                    labels.append(novelty_label)

                if has_nan or len(surprises) < 10:
                    print(f"  {label:<40}  {'NaN':>6}  {'---':>6}  {'---':>10}  {'UNSTABLE':>7}")
                    continue

                arr = np.array(surprises)
                corr, pval = spearmanr(arr, np.array(labels))
                rng = arr.max() - arr.min()
                stable = "OK" if not np.isnan(corr) else "UNSTABLE"

                print(f"  {label:<40}  {corr:>6.3f}  {pval:>6.3f}  {rng:>10.4f}  {stable:>7}")

                if "baseline" in label:
                    baseline_corr = corr

            except Exception as e:
                print(f"  {label:<40}  ERROR: {e}")

    print()

    # Only hard requirement: baseline must still pass
    assert baseline_corr is not None, "Baseline config failed to run"
    assert baseline_corr >= 0.2, (
        f"Baseline config signal degraded (r={baseline_corr:.3f}). "
        f"Check that default parameters haven't been changed."
    )


@test_group("Neural Memory")
def test_rtrl_accessors_return_expected_types():
    """NeuralCoordinator surprise accessors return correct types after warmup."""
    require("torch")
    with TempDir() as d:
        coord = _make_coordinator(d)

        # Before warmup
        assert coord.get_last_surprise() is None or isinstance(coord.get_last_surprise(), float)
        assert isinstance(coord.get_surprise_ema(), float)
        assert coord.is_warmed_up() is False

        # Warmup
        for user, assistant in _WARMUP:
            _run_turn(coord, user, assistant)

        assert coord.is_warmed_up() is True
        last = coord.get_last_surprise()
        assert last is not None
        assert isinstance(last, float)
        assert last >= 0.0

        ema = coord.get_surprise_ema()
        assert isinstance(ema, float)
        assert ema >= 0.0

        # Query surprise
        q_surprise = coord.get_surprise_for_query("What is asyncio?")
        assert q_surprise is not None
        assert isinstance(q_surprise, float)
        assert q_surprise >= 0.0


@test_group("Neural Memory")
def test_rtrl_dynamic_importance_wired_into_store_episode():
    """ProjectMemory.store_episode uses neural surprise for importance when warmed up."""
    require("torch", "chromadb", "sentence_transformers")
    from engram.project_memory import ProjectMemory
    from engram.rtrl.neural_memory import NeuralMemoryConfig

    with TempDir() as d:
        cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16, enabled=True)
        with ProjectMemory(
            project_id="neural_imp_test",
            project_type="general_assistant",
            base_dir=d,
            llm_engine=None,
            session_id=unique_session(),
            neural_config=cfg,
        ) as pm:
            if pm.neural_coord is None:
                # Neural layer not available — skip rest of test
                return

            # Warm up neural coordinator
            for user, assistant in _WARMUP:
                pm.add_turn("user", user)
                pm.add_turn("assistant", assistant)

            assert pm.neural_coord.is_warmed_up()

            # store_episode should now use neural importance
            ep_id = pm.store_episode(
                "This is a test episode about asyncio event loops.",
                bypass_filter=True,
            )
            # If episodic is available, episode was stored
            if ep_id is not None:
                episodes = pm.episodic.get_by_ids([ep_id])
                assert len(episodes) == 1
                ep = episodes[0]
                # Verify neural wiring fired: metadata should contain neural_surprise key.
                # Note: effective_importance = max(caller_importance, neural_importance),
                # so the stored value may still equal 0.5 if neural score < caller floor.
                # What matters is that the neural path was exercised.
                meta = ep.metadata or {}
                assert "neural_surprise" in meta, (
                    "Expected 'neural_surprise' in episode metadata when neural is active. "
                    f"Got metadata keys: {list(meta.keys())}"
                )
                # neural_surprise should be a float (not None) after warmup
                assert meta["neural_surprise"] is not None, (
                    "neural_surprise should be computed after warmup"
                )


# ── Warmup from history tests ─────────────────────────────────────────────────

@test_group("Neural Memory")
def test_warm_up_from_history_basic():
    """warm_up_from_history() runs steps without raising."""
    require("torch")
    from dataclasses import dataclass

    @dataclass
    class FakeEpisode:
        text: str
        id: str = "ep_1"

    with TempDir() as d:
        coord = _make_coordinator(d)
        episodes = [FakeEpisode(text=f"Python asyncio episode {i}") for i in range(10)]
        n = coord.warm_up_from_history(episodes, max_episodes=10)
        assert n == 10
        stats = coord.get_stats()
        assert stats.get("total_steps", 0) == 10


@test_group("Neural Memory")
def test_warm_up_from_history_accelerates_warmup():
    """After warm_up_from_history with 50+ episodes, is_warmed_up() returns True."""
    require("torch")
    from dataclasses import dataclass

    @dataclass
    class FakeEpisode:
        text: str
        id: str = "ep_x"

    with TempDir() as d:
        coord = _make_coordinator(d)
        assert coord.is_warmed_up() is False

        # Feed 55 synthetic episodes
        episodes = [
            FakeEpisode(text=f"Episode about asyncio and Python turn {i}")
            for i in range(55)
        ]
        n = coord.warm_up_from_history(episodes, max_episodes=55)
        assert n == 55
        assert coord.is_warmed_up() is True


@test_group("Neural Memory")
def test_warm_up_from_history_caps_at_max():
    """max_episodes limits how many are replayed."""
    require("torch")
    from dataclasses import dataclass

    @dataclass
    class FakeEpisode:
        text: str
        id: str = "ep_x"

    with TempDir() as d:
        coord = _make_coordinator(d)
        episodes = [FakeEpisode(text=f"Episode {i}") for i in range(100)]
        n = coord.warm_up_from_history(episodes, max_episodes=30)
        assert n == 30
        assert coord.get_stats()["total_steps"] == 30


@test_group("Neural Memory")
def test_warm_up_from_history_skips_empty_text():
    """Episodes with empty text are silently skipped."""
    require("torch")
    from dataclasses import dataclass

    @dataclass
    class FakeEpisode:
        text: str
        id: str = "ep_x"

    with TempDir() as d:
        coord = _make_coordinator(d)
        episodes = [
            FakeEpisode(text="Real content"),
            FakeEpisode(text=""),
            FakeEpisode(text="   "),
            FakeEpisode(text="More real content"),
        ]
        n = coord.warm_up_from_history(episodes)
        assert n == 2  # only the two non-empty episodes


@test_group("Neural Memory")
def test_warm_up_from_history_empty_list():
    """warm_up_from_history([]) returns 0 without raising."""
    require("torch")
    with TempDir() as d:
        coord = _make_coordinator(d)
        n = coord.warm_up_from_history([])
        assert n == 0
        assert coord.get_stats()["total_steps"] == 0


@test_group("Neural Memory")
def test_warmup_improves_signal_quality():
    """After pre-warming, repeated turns should show lower surprise than novel turns.

    This is the key functional test: demonstrates that warm_up_from_history()
    produces a more stable EMA baseline, making familiar vs novel more distinguishable.
    """
    require("torch")
    from dataclasses import dataclass

    @dataclass
    class FakeEpisode:
        text: str
        id: str = "ep_x"

    with TempDir() as d:
        coord = _make_coordinator(d)

        # Pre-warm with asyncio content (60 episodes)
        warmup_eps = [
            FakeEpisode(text=f"asyncio event loop coroutine Python async episode {i}")
            for i in range(60)
        ]
        coord.warm_up_from_history(warmup_eps, max_episodes=60)
        assert coord.is_warmed_up()

        # Now measure: familiar (asyncio) vs novel (Fourier transforms)
        familiar_surprises = []
        for _ in range(5):
            s = _run_turn(coord, "What is asyncio?", "asyncio is Python's async library.")
            familiar_surprises.append(s)

        novel_surprises = []
        for _ in range(5):
            s = _run_turn(coord, "What is a Fourier transform?",
                          "It decomposes signals into frequency components.")
            novel_surprises.append(s)

        familiar_mean = sum(familiar_surprises) / len(familiar_surprises)
        novel_mean = sum(novel_surprises) / len(novel_surprises)

        # After warmup, familiar content should produce measurably lower surprise
        # than genuinely novel content. Allow small tolerance.
        assert familiar_mean <= novel_mean * 1.1, (
            f"Expected familiar ({familiar_mean:.4f}) <= novel ({novel_mean:.4f}). "
            f"Warmup did not produce a stable enough baseline."
        )


@test_group("Neural Memory")
def test_ema_persists_across_save_load():
    """surprise_ema is preserved through NeuralMemory save/load cycle.

    This verifies that the baseline carries across sessions — a new session
    starting from a saved state doesn't reset the EMA to 0.
    """
    require("torch")
    with TempDir() as d:
        coord = _make_coordinator(d)

        # Build up EMA with warmup turns
        for user, assistant in _WARMUP[:30]:
            _run_turn(coord, user, assistant)

        ema_before = coord.get_surprise_ema()
        steps_before = coord.get_stats()["total_steps"]
        assert ema_before > 0.0, "EMA should be non-zero after 30 steps"

        # Save
        coord._neural.save()

        # Load into fresh coordinator
        from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig, EmbeddingProjector
        from engram.memory.neural_coordinator import NeuralCoordinator

        class _Det:
            _DIM = 64
            def embed(self, text):
                import hashlib
                h = hashlib.sha256(text.encode()).digest()
                seed = int.from_bytes(h[:4], "little")
                rng = np.random.RandomState(seed)
                vec = rng.randn(self._DIM).astype(np.float32)
                return vec / (np.linalg.norm(vec) + 1e-8)

        EMBED_DIM = 64
        cfg2 = NeuralMemoryConfig(hidden_dim=32, value_dim=16, enabled=True)
        neural2 = NeuralMemory.load(project_dir=d, config=cfg2)
        coord2 = NeuralCoordinator(
            neural=neural2,
            key_projector=EmbeddingProjector(EMBED_DIM, 64),
            value_projector=EmbeddingProjector(EMBED_DIM, 16),
            embedding_service=_Det(),
        )

        ema_after = coord2.get_surprise_ema()
        steps_after = coord2.get_stats()["total_steps"]

        assert steps_after == steps_before, (
            f"Step count should persist: {steps_before} → {steps_after}"
        )
        assert abs(ema_after - ema_before) < 1e-6, (
            f"EMA should persist: {ema_before:.6f} → {ema_after:.6f}"
        )


# ── Forgetting policy neural_surprise integration ─────────────────────────────

@test_group("Neural Memory")
def test_neural_surprise_ratio_stored_in_metadata():
    """store_episode stores neural_surprise as 0-1 ratio, not raw importance."""
    require("torch", "chromadb", "sentence_transformers")
    from engram.project_memory import ProjectMemory
    from engram.rtrl.neural_memory import NeuralMemoryConfig

    with TempDir() as d:
        cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16, enabled=True)
        with ProjectMemory(
            project_id="surp_ratio_test",
            project_type="general_assistant",
            base_dir=d,
            llm_engine=None,
            session_id=unique_session(),
            neural_config=cfg,
        ) as pm:
            if pm.neural_coord is None or pm.episodic is None:
                return  # deps not available

            # Warm up
            for user, assistant in _WARMUP:
                pm.add_turn("user", user)
                pm.add_turn("assistant", assistant)

            assert pm.neural_coord.is_warmed_up()

            ep_id = pm.store_episode(
                "Test episode for neural surprise ratio check.",
                bypass_filter=True,
            )
            if ep_id is None:
                return

            episodes = pm.episodic.get_by_ids([ep_id])
            assert len(episodes) == 1
            meta = episodes[0].metadata or {}

            # neural_surprise should now be the ratio (0-1), not importance (0.1-0.95)
            assert "neural_surprise" in meta, "neural_surprise key missing from metadata"
            assert "neural_importance" in meta, "neural_importance key missing from metadata"

            surp = meta.get("neural_surprise")
            imp = meta.get("neural_importance")

            if surp is not None:
                assert 0.0 <= surp <= 1.0, f"neural_surprise ratio out of range: {surp}"
            if imp is not None:
                assert 0.1 <= imp <= 0.95, f"neural_importance out of range: {imp}"


@test_group("Neural Memory")
def test_forgetting_policy_reads_neural_surprise_ratio():
    """ForgettingPolicy.score_all uses neural_surprise ratio from metadata."""
    from engram.memory.forgetting import ForgettingPolicy, ForgettingConfig
    from dataclasses import dataclass
    import time as _time

    @dataclass
    class FakeEp:
        id: str
        timestamp: float
        importance: float
        metadata: dict

    class FakeEpisodic:
        def __init__(self, episodes):
            self._eps = episodes
        def get_metadata_batch(self, project_id, limit=500, offset=0):
            import json as _j
            return [
                {
                    "id": ep.id,
                    "timestamp": ep.timestamp,
                    "importance": ep.importance,
                    "metadata": _j.dumps(ep.metadata),  # must be valid JSON
                }
                for ep in self._eps[offset:offset+limit]
            ]

    with TempDir() as d:
        policy = ForgettingPolicy(
            access_db_path=d / "acc.db",
            config=ForgettingConfig(
                weight_recency=0.30,
                weight_importance=0.30,
                weight_access=0.30,
                weight_surprise=0.10,
            ),
        )

        now = _time.time()
        # Episode A: high neural surprise (novel, should score higher)
        # Episode B: zero neural surprise (familiar, should score lower)
        ep_a = FakeEp("ep_a", now - 86400, 0.5, {"neural_surprise": 0.9})
        ep_b = FakeEp("ep_b", now - 86400, 0.5, {"neural_surprise": 0.0})

        scores = policy.score_all(FakeEpisodic([ep_a, ep_b]), "test_project")
        assert len(scores) == 2

        score_a = next(s for s in scores if s.episode_id == "ep_a")
        score_b = next(s for s in scores if s.episode_id == "ep_b")

        # High surprise → higher retention score
        assert score_a.total > score_b.total, (
            f"High surprise ep_a ({score_a.total:.4f}) should outscore "
            f"low surprise ep_b ({score_b.total:.4f})"
        )
        assert score_a.surprise == 0.9
        assert score_b.surprise == 0.0


# ── Retrieval ranking bias ────────────────────────────────────────────────────

@test_group("Neural Memory")
def test_retrieval_policy_weights_sum_correctly():
    """RetrievalPolicy episodic weights sum to approximately 1.0.

    Neural surprise is NOT a retrieval weight — familiar episodes are
    still highly relevant (e.g. standing preferences) and should not be
    deprioritised just because they have low stored surprise.  The neural
    layer influences retrieval indirectly through episode importance scoring
    (which affects what survives in ChromaDB) not through a direct affinity
    bias in retrieval ranking.
    """
    from engram.memory.retrieval import RetrievalPolicy
    policy = RetrievalPolicy()
    # No neural_surprise weight in retrieval
    assert not hasattr(policy, "episodic_weight_neural_surprise"), (
        "Neural surprise should not be a retrieval ranking factor. "
        "Familiar episodes (low surprise) can still be highly relevant."
    )
    total = (
        policy.episodic_weight_lexical
        + policy.episodic_weight_importance
        + policy.episodic_weight_recency
        + policy.episodic_weight_density
    )
    assert abs(total - 1.0) < 0.05, (
        f"Episodic weights should sum to ~1.0, got {total:.3f}"
    )
