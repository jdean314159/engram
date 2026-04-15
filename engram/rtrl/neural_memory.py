"""
Neural Memory Layer — Sub-symbolic adaptation via RTRL.

Fifth layer of Engram's memory architecture. Uses a subgrouped
RTRL network with TITANS surprise gating for continuous online
learning during inference.

Unlike the text-level layers (working, episodic, semantic), neural
memory operates in embedding space — it learns implicit patterns
from external text embeddings and deterministic projections, not from
direct hidden-state hooks into the serving backend.

Key characteristics:
    - 0.3ms per step (CPU), ~5K parameters
    - Learns user-specific patterns over sessions
    - Surprise-gated: updates only on novel information
    - Persistent: saves/loads learned weights per project
    - Pluggable: enable/disable per project via config

Optimal parameters (from parameter sweep):
    hidden_dim=32, value_dim=16
    leaky_relu/linear activations
    lr=0.003, grad_clip_norm=1.0

    Note: grad_clip_norm=1.0 is required at hidden_dim=32.
    The prior value of 5.0 causes P-matrix overflow at this
    dimension. lr=0.001 is a stable alternative for sessions
    exceeding ~1000 turns.

Author: Jeffrey Dean
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .core import TITANSConfig, TITANSMemory

logger = logging.getLogger(__name__)


class EmbeddingProjector:
    """Projects high-dimensional text embeddings to RTRL-compatible dimensions.

    Uses a seeded random projection matrix (Johnson-Lindenstrauss).
    Deterministic per seed so projections are consistent across sessions.

    The all-MiniLM-L6-v2 model used by episodic memory produces 384-dim
    embeddings. RTRL operates on key_dim=64, value_dim=16. This bridges
    the gap without requiring a learned projection.
    """

    def __init__(self, input_dim: int, output_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        rng = np.random.RandomState(seed)
        # Scaled for approximate distance preservation
        self.matrix = rng.randn(output_dim, input_dim).astype(np.float32)
        self.matrix /= np.sqrt(output_dim)

    def __call__(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding to lower dimension."""
        return self.matrix @ embedding.astype(np.float32)


@dataclass
class NeuralMemoryConfig:
    """Configuration for neural memory layer.

    Defaults reflect parameter sweep results:
      hidden_dim=32, lr=0.003, grad_clip_norm=1.0, gated=True

    Key findings from sweep (3-pair associative memory benchmark):
      - hidden_dim=32 is the capacity sweet spot: larger dims (64+) cause
        P-matrix overflow even with clipping; smaller dims (8-16) converge
        but reach higher final surprise.
      - lr=0.003 gives best convergence for the multi-pair case; lr=0.001
        is marginally slower but more stable in very long sessions (1000+
        turns).  0.003 is the production default.
      - grad_clip_norm=1.0 is necessary to prevent P-matrix accumulation
        from overflowing at hidden_dim >= 32.  The prior thesis value of
        5.0 was calibrated for a smaller hidden_dim and should not be used.
      - leaky_relu/linear activations outperform tanh/linear for embedding
        regression (preserves gradient sign through dead zones).
    """
    enabled: bool = True

    # Dimensions
    key_dim: int = 64           # Input embedding dimension
    value_dim: int = 16         # Output dimension
    hidden_dim: int = 32        # RTRL hidden neurons (do not increase above 32)

    # Embedding projection (text embedder → RTRL dimensions)
    embedding_dim: int = 384    # all-MiniLM-L6-v2 output dimension
    projection_seed: int = 42   # Deterministic projection per project

    # Learning — optimal from sweep
    lr: float = 0.003           # 0.001 is stable alternative for very long sessions
    grad_clip_norm: float = 1.0 # Must be ≤ 1.0 for hidden_dim=32; 5.0 causes overflow
    weight_decay: float = 1e-5
    surprise_threshold: float = 0.001
    surprise_modulated_lr: bool = True

    # Architecture (do not change without re-sweeping)
    gated: bool = True          # GRU gating — non-optional
    layer_norm: bool = False
    hidden_activation: str = "leaky_relu"
    output_activation: str = "linear"
    optimizer: str = "adam"
    init: str = "xavier"

    # Persistence
    save_hidden_state: bool = True  # Save full state for exact resume

    # Runtime
    device: str = "cpu"         # RTRL is fast enough on CPU
    dtype: str = "float32"
    verbose: bool = False

    # Identity / compatibility
    model_fingerprint: str = "default"


class NeuralMemory:
    """Sub-symbolic memory layer using RTRL with TITANS surprise gating.

    Wraps TITANSMemory with per-project persistence, metrics tracking,
    and a consistent interface matching Engram's other memory layers.

    Usage:
        nmem = NeuralMemory(project_dir=Path("./data/my_project"))

        # During inference: process an embedding pair
        result = nmem.step(key_embedding, value_embedding)
        # result = {
        #     'prediction': np.array(...),  # What memory expected
        #     'surprise': 0.042,            # How novel this was
        #     'written': True,              # Whether memory updated
        #     'step': 147,                  # Total steps processed
        # }

        # Read without updating
        prediction = nmem.read(query_embedding)

        # Persist between sessions
        nmem.save()
        # ...later...
        nmem = NeuralMemory.load(project_dir)
    """

    FILENAME = "neural_memory.json"

    def __init__(
        self,
        project_dir: Optional[Path] = None,
        config: Optional[NeuralMemoryConfig] = None,
    ):
        """Initialize neural memory.

        Args:
            project_dir: Directory for persistence. None = no auto-save.
            config: Configuration. None = optimal defaults.
        """
        self.config = config or NeuralMemoryConfig()
        self.project_dir = Path(project_dir).expanduser().resolve(strict=False) if project_dir else None
        self._memory: Optional[TITANSMemory] = None
        self._session_start = time.time()
        self._memory_role = "auxiliary_embedding_memory"
        self._session_steps = 0

        if self.config.enabled:
            self._init_memory()

    def _init_memory(self):
        """Create or load the underlying TITANSMemory."""
        # Check for saved state
        save_path = self._save_path()
        if save_path and save_path.exists():
            try:
                self._memory = TITANSMemory.load(str(save_path))
                logger.info("Loaded neural memory from %s (%d prior steps)",
                            save_path, self._memory._step_count)
                return
            except Exception as e:
                logger.warning("Failed to load neural memory: %s. Starting fresh.", e)

        # Create fresh
        tcfg = TITANSConfig(
            key_dim=self.config.key_dim,
            value_dim=self.config.value_dim,
            hidden_dim=self.config.hidden_dim,
            lr=self.config.lr,
            grad_clip_norm=self.config.grad_clip_norm,
            weight_decay=self.config.weight_decay,
            surprise_threshold=self.config.surprise_threshold,
            surprise_modulated_lr=self.config.surprise_modulated_lr,
            gated=self.config.gated,
            layer_norm=self.config.layer_norm,
            hidden_activation=self.config.hidden_activation,
            output_activation=self.config.output_activation,
            optimizer=self.config.optimizer,
            init=self.config.init,
            device=self.config.device,
            dtype=self.config.dtype,
            verbose=self.config.verbose,
        )
        self._memory = TITANSMemory(tcfg)
        logger.info("Created fresh neural memory: %d params",
                     self._param_count())

    def _save_path(self) -> Optional[Path]:
        if self.project_dir:
            return self.project_dir / self.FILENAME
        return None

    def _param_count(self) -> int:
        """Count total trainable parameters."""
        if self._memory is None:
            return 0
        net = self._memory.net
        B = net.B
        count = B.numel(net.weights)
        if net.config.gated:
            count += B.numel(net.gate_weights) + B.numel(net.gate_bias)
        if net.config.layer_norm:
            count += B.numel(net.ln_gamma) + B.numel(net.ln_beta)
        return count

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def read(self, query: np.ndarray) -> np.ndarray:
        """Read from memory without updating.

        Args:
            query: Key vector, shape (key_dim,)

        Returns:
            Predicted value vector, shape (value_dim,)
        """
        if not self.config.enabled or self._memory is None:
            return np.zeros(self.config.value_dim)
        return self._memory.read(query)

    def step(self, key: np.ndarray, value: np.ndarray) -> Dict[str, Any]:
        """Process a key-value pair: read prediction, then conditionally write.

        This is the primary interface during LLM inference.

        Args:
            key: Input embedding, shape (key_dim,)
            value: Target embedding, shape (value_dim,)

        Returns:
            Dict with 'predicted', 'surprise', 'wrote', etc.
        """
        if not self.config.enabled or self._memory is None:
            return {
                'predicted': np.zeros(self.config.value_dim),
                'surprise': 0.0,
                'wrote': False,
                'effective_lr': 0.0,
                'surprise_ema': 0.0,
            }

        result = self._memory.step(key, value)
        self._session_steps += 1
        return result

    def surprise(self, key: np.ndarray, value: np.ndarray) -> float:
        """Compute surprise without updating memory.

        Useful for evaluating how novel a pattern is.
        """
        if not self.config.enabled or self._memory is None:
            return 0.0
        return self._memory.surprise(key, value)

    def forget(self, decay: Optional[float] = None):
        """Apply explicit weight decay to forget old patterns."""
        if self._memory is not None:
            self._memory.forget(decay)

    def reset(self):
        """Reset hidden state but keep learned weights.

        Use between sessions when context changes but knowledge should persist.
        """
        if self._memory is not None:
            self._memory.reset()
            self._session_steps = 0
            self._session_start = time.time()

    def reset_full(self):
        """Reset everything — weights and hidden state.

        Use after a meaningful model or embedding change when old compensatory
        weights are no longer valid.
        """
        if self._memory is not None:
            self._memory.reset_full()
            self._session_steps = 0
            self._session_start = time.time()

    def ensure_compatible(self, model_fingerprint: str) -> bool:
        """Reset full state when the serving/embedding identity changes."""
        current = getattr(self.config, "model_fingerprint", "default")
        if current == model_fingerprint:
            return False
        logger.info("Neural memory fingerprint changed from %s to %s; resetting state", current, model_fingerprint)
        self.config.model_fingerprint = model_fingerprint
        self.reset_full()
        return True

    def describe_role(self) -> Dict[str, Any]:
        return {
            "memory_role": self._memory_role,
            "backend_agnostic": True,
            "model_fingerprint": getattr(self.config, "model_fingerprint", "default"),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None):
        """Save memory state to disk.

        Args:
            path: Override save path. None = use project_dir.
        """
        if self._memory is None:
            return

        save_path = Path(path).expanduser().resolve(strict=False) if path else self._save_path()
        if save_path is None:
            logger.warning("No save path configured for neural memory")
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory.save(
            str(save_path),
            include_hidden_state=self.config.save_hidden_state,
        )
        logger.info("Saved neural memory to %s", save_path)

    @classmethod
    def load(
        cls,
        project_dir: Path,
        config: Optional[NeuralMemoryConfig] = None,
    ) -> 'NeuralMemory':
        """Load neural memory from project directory.

        Args:
            project_dir: Project directory containing neural_memory.json.
            config: Override config. None = use saved config.
        """
        nmem = cls(project_dir=project_dir, config=config)
        # _init_memory already tried to load from save_path
        return nmem

    # ------------------------------------------------------------------
    # Metrics & stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if self._memory is None:
            return {"enabled": False}

        mem_stats = self._memory.stats.copy()
        total = mem_stats['total_writes'] + mem_stats['total_skipped']

        return {
            "enabled": self.config.enabled,
            "memory_role": self._memory_role,
            "backend_agnostic": True,
            "model_fingerprint": getattr(self.config, "model_fingerprint", "default"),
            "total_steps": self._memory._step_count,
            "session_steps": self._session_steps,
            "total_writes": mem_stats['total_writes'],
            "total_skipped": mem_stats['total_skipped'],
            "write_ratio": self._memory.write_ratio,
            "avg_surprise": self._memory.avg_surprise,
            "max_surprise": mem_stats['max_surprise'],
            "params": self._param_count(),
            "session_duration_s": time.time() - self._session_start,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        if self._memory is None:
            return "NeuralMemory: disabled"
        return self._memory.summary()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self):
        """Save and release resources."""
        if self.config.enabled:
            self.save()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        steps = self._memory._step_count if self._memory else 0
        return (
            f"NeuralMemory(key_dim={self.config.key_dim}, "
            f"value_dim={self.config.value_dim}, "
            f"steps={steps})"
        )
