"""
Modern Subgrouped RTRL Neural Network
=======================================
PyTorch/NumPy implementation of Dean (1994) subgrouped RTRL,
modernized with Adam, gating, layer norm, and a TITANS-compatible
memory interface.

Stages:
  1. Core RTRL with vectorized P matrix, Xavier init, tanh/sigmoid
  2. Adam optimizer, gradient clipping, LR scheduling
  3. GRU-style update gate, pre-activation layer normalization
  4. TITANSMemory — surprise-gated associative memory for LLM integration

Usage (direct RTRL):
    cfg = RTRLConfig(num_inputs=2, num_outputs=1, num_hidden=4)
    net = ModernSubgroupedRTRL(cfg)
    history = net.train(data, targets)

Usage (TITANS memory):
    mem = TITANSMemory(key_dim=64, value_dim=64, hidden_dim=32)
    result = mem.step(key_vector, value_vector)
    retrieved = mem.read(query_vector)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import math
import time
import json

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Backend abstraction: unified API over torch or numpy
# ---------------------------------------------------------------------------

class _Backend:
    """Thin wrapper so the RTRL class doesn't branch on every operation."""

    def __init__(self, use_torch: bool, device: str, dtype_str: str):
        self.use_torch = use_torch and HAS_TORCH
        if self.use_torch:
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            self.dtype = getattr(torch, dtype_str)
            self.device_name = str(self.device)
        else:
            self.device = None
            self.dtype = getattr(np, dtype_str)
            self.device_name = "cpu(numpy)"

    def zeros(self, *shape):
        if self.use_torch:
            return torch.zeros(*shape, dtype=self.dtype, device=self.device)
        return np.zeros(shape, dtype=self.dtype)

    def randn(self, *shape):
        if self.use_torch:
            return torch.randn(*shape, dtype=self.dtype, device=self.device)
        return np.random.randn(*shape).astype(self.dtype)

    def uniform(self, low, high, shape):
        if self.use_torch:
            return torch.empty(shape, dtype=self.dtype, device=self.device).uniform_(low, high)
        return np.random.uniform(low, high, shape).astype(self.dtype)

    def eye(self, n):
        if self.use_torch:
            return torch.eye(n, dtype=self.dtype, device=self.device)
        return np.eye(n, dtype=self.dtype)

    def arange(self, start, stop, step):
        if self.use_torch:
            return torch.arange(start, stop, step, device=self.device)
        return np.arange(start, stop, step)

    def zeros_like(self, x):
        if self.use_torch:
            return torch.zeros_like(x)
        return np.zeros_like(x)

    def clone(self, x):
        if self.use_torch:
            return x.clone()
        return x.copy()

    def from_numpy(self, arr):
        if self.use_torch:
            return torch.from_numpy(arr).to(dtype=self.dtype, device=self.device)
        return arr.astype(self.dtype) if arr.dtype != self.dtype else arr

    def to_numpy(self, x):
        if self.use_torch:
            return x.detach().cpu().numpy()
        return x

    def sigmoid(self, x):
        if self.use_torch:
            return torch.sigmoid(x)
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        if self.use_torch:
            return torch.tanh(x)
        return np.tanh(np.clip(x, -50, 50))

    def relu(self, x):
        if self.use_torch:
            return torch.relu(x).clamp(max=6.0)
        return np.clip(np.maximum(0, x), 0, 6.0)

    def leaky_relu(self, x, slope=0.01):
        if self.use_torch:
            return F.leaky_relu(x, slope).clamp(-6.0, 6.0)
        return np.clip(np.where(x > 0, x, slope * x), -6.0, 6.0)

    def clamp(self, x, mn=None, mx=None):
        if self.use_torch:
            return x.clamp(min=mn, max=mx)
        return np.clip(x, mn, mx)

    def matmul(self, a, b):
        if self.use_torch:
            return a @ b
        return a @ b

    def mv(self, mat, vec):
        if self.use_torch:
            return torch.mv(mat, vec)
        return mat @ vec

    def argmax(self, x):
        if self.use_torch:
            return x.argmax().item()
        return int(np.argmax(x))

    def sum(self, x):
        if self.use_torch:
            return x.sum().item()
        return float(np.sum(x))

    def numel(self, x):
        if self.use_torch:
            return x.numel()
        return x.size

    def sqrt(self, x):
        if self.use_torch:
            return torch.sqrt(x)
        return np.sqrt(x)

    def norm(self, x):
        """Frobenius / L2 norm of entire tensor."""
        if self.use_torch:
            return x.norm().item()
        return float(np.linalg.norm(x.ravel()))


@dataclass
class RTRLConfig:
    """Configuration for Modern Subgrouped RTRL."""
    # Architecture
    num_inputs: int
    num_outputs: int
    num_hidden: int
    time_delay: int = 2

    # Activations
    hidden_activation: str = "tanh"     # tanh | sigmoid | relu | leaky_relu
    output_activation: str = "sigmoid"  # sigmoid | tanh | linear

    # Training
    epochs: int = 100
    lr: float = 0.01
    momentum: float = 0.0
    per_step_dw_reset: bool = True      # True = thesis C behavior

    # Optimizer: "sgd" (thesis) or "adam"
    optimizer: str = "sgd"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0           # L2 regularization

    # Gradient clipping (0 = disabled)
    grad_clip_norm: float = 0.0

    # LR schedule: "fixed" | "plateau" | "cosine" | "thesis"
    lr_schedule: str = "fixed"
    plateau_patience: int = 10
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-7
    cosine_T_max: int = 0               # 0 = use epochs
    cosine_eta_min: float = 1e-6

    # Initialization
    init: str = "xavier"                # xavier | he | thesis (uniform[-1,1])

    # Architecture options
    gated: bool = False                 # GRU-style update gate per neuron
    layer_norm: bool = False            # Pre-activation layer normalization
    ln_eps: float = 1e-5                # Layer norm epsilon

    # RTRL parameters
    y_prime_min: float = 0.01
    teacher_forcing: bool = False
    categorical_output: bool = True
    continuous_epochs: bool = False
    skip_threshold: float = 0.0
    ok_threshold: float = 0.125
    max_activation: float = 50.0

    # Legacy thesis LR decay (use lr_schedule="thesis" to enable)
    lr_decay_factor: float = 0.1
    lr_decay_threshold: float = 1.01
    lr_min: float = 1e-9

    # Runtime
    device: str = "auto"                # auto | cuda | cpu
    dtype: str = "float32"
    verbose: bool = True

    def __post_init__(self):
        if self.num_hidden % self.num_outputs != 0:
            self.num_hidden = ((self.num_hidden // self.num_outputs) + 1) * self.num_outputs
            if self.verbose:
                print(f"Adjusted hidden nodes to {self.num_hidden} for equal subgrouping")


class ModernSubgroupedRTRL:
    """
    Modern Subgrouped RTRL with PyTorch/NumPy backend.

    Architecture (from thesis):
        N = num_hidden + num_outputs neurons, fully recurrent
        G = num_outputs subgroups, each of size S = N/G
        First neuron per subgroup = output neuron
        P matrix: [S, ncols, N] — subgrouped Jacobian

    Stage 1: Core RTRL, Xavier init, tanh/sigmoid, vectorized P matrix
    Stage 2: Adam optimizer, gradient clipping, LR scheduling
    Stage 3: GRU-style update gate, pre-activation layer normalization
    """

    def __init__(self, config: RTRLConfig):
        self.config = config

        # Backend (torch if available and requested, else numpy)
        use_torch = (config.device != "cpu") or HAS_TORCH
        self.B = _Backend(use_torch, config.device, config.dtype)

        # Dimensions (same naming as thesis C code)
        self.m = config.num_inputs + 1          # bias + external inputs
        self.n = config.num_hidden + config.num_outputs
        self.ncols = self.m + self.n            # total input width
        self.nrows = self.n                     # total neurons
        self.num_groups = config.num_outputs
        self.group_size = self.n // self.num_groups

        # Initialize
        self._init_weights()
        self._init_state()
        self._build_index_cache()
        self._init_optimizer()
        self._init_scheduler()

        # History
        self.training_history: List[Dict[str, Any]] = []

        if config.verbose:
            n_params = self.B.numel(self.weights)
            extras = []
            if config.gated:
                n_params += self.B.numel(self.gate_weights) + self.B.numel(self.gate_bias)
                extras.append("gated")
            if config.layer_norm:
                n_params += self.B.numel(self.ln_gamma) + self.B.numel(self.ln_beta)
                extras.append("LN")
            extra_str = f" [{'+'.join(extras)}]" if extras else ""
            print(f"ModernSubgroupedRTRL({self.B.device_name}) "
                  f"n={self.n} groups={self.num_groups} gsize={self.group_size} "
                  f"params={n_params}{extra_str} "
                  f"P={list(self.p_matrix.shape)}")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self):
        """Initialize weight matrix."""
        shape = (self.nrows, self.ncols)
        cfg = self.config
        B = self.B

        if cfg.init == "xavier":
            std = math.sqrt(2.0 / (self.ncols + self.nrows))
            self.weights = B.randn(*shape) * std
        elif cfg.init == "he":
            std = math.sqrt(2.0 / self.ncols)
            self.weights = B.randn(*shape) * std
        elif cfg.init == "thesis":
            self.weights = B.uniform(-1, 1, shape)
        else:
            self.weights = B.uniform(-1, 1, shape)

        self.delta_weights = B.zeros_like(self.weights)

        # GRU-style gate weights: gate = σ(W_gate @ z + b_gate)
        if cfg.gated:
            std_g = math.sqrt(2.0 / (self.ncols + self.nrows))
            self.gate_weights = B.randn(self.nrows, self.ncols) * std_g
            self.gate_bias = B.zeros(self.nrows)
            # Initialize bias positive so gates start ~open (pass new info)
            self.gate_bias[:] = -1.0  # σ(-1) ≈ 0.27 → mostly use new activation

        # Layer normalization parameters
        if cfg.layer_norm:
            if B.use_torch:
                self.ln_gamma = torch.ones(self.nrows, dtype=B.dtype, device=B.device)
                self.ln_beta = B.zeros(self.nrows)
            else:
                self.ln_gamma = np.ones(self.nrows, dtype=B.dtype)
                self.ln_beta = B.zeros(self.nrows)

    def _init_state(self):
        """Initialize P matrices and network state."""
        B = self.B
        self.p_matrix = B.zeros(self.group_size, self.ncols, self.nrows)
        self.p_matrix_old = B.zeros(self.group_size, self.ncols, self.nrows)
        self.outputs = B.zeros(self.nrows)
        self.activations = B.zeros(self.nrows)
        self.errors = B.zeros(self.num_groups)
        # Gate state (stored for P matrix update)
        if self.config.gated:
            self.gate_values = B.zeros(self.nrows)
            self.prev_outputs = B.zeros(self.nrows)
        # LN state (stored for P matrix derivative)
        if self.config.layer_norm:
            self.ln_std = B.zeros(1)  # scalar, stored for derivative

    def _build_index_cache(self):
        """Pre-compute index arrays."""
        self.output_indices = list(range(0, self.n, self.group_size))
        self._eye = self.B.eye(self.group_size)

    def _init_optimizer(self):
        """Initialize optimizer state (Adam or SGD)."""
        B = self.B
        if self.config.optimizer == "adam":
            # Adam first and second moment estimates
            self.adam_m = B.zeros_like(self.weights)    # first moment
            self.adam_v = B.zeros_like(self.weights)    # second moment
            self.adam_t = 0                              # timestep counter
        # Gradient buffer (always needed — raw gradient before optimizer)
        self.grad_buffer = B.zeros_like(self.weights)

        # Gate optimizer (always Adam, separate from main weights)
        if self.config.gated:
            self.gate_adam_m_w = B.zeros_like(self.gate_weights)
            self.gate_adam_v_w = B.zeros_like(self.gate_weights)
            self.gate_adam_m_b = B.zeros_like(self.gate_bias)
            self.gate_adam_v_b = B.zeros_like(self.gate_bias)
            self.gate_adam_t = 0

    def _init_scheduler(self):
        """Initialize LR scheduler state."""
        self._base_lr = self.config.lr
        self._current_lr = self.config.lr
        # Plateau scheduler state
        self._plateau_best = float('inf')
        self._plateau_wait = 0

    # ------------------------------------------------------------------
    # State snapshot / restore (for non-mutating read and surprise)
    # ------------------------------------------------------------------

    def _clone_scalar_or_tensor(self, value):
        if value is None:
            return None
        if isinstance(value, (int, float, bool, str)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        return self.B.clone(value)

    def _restore_state_value(self, value):
        if value is None:
            return None
        if isinstance(value, list):
            return self.B.from_numpy(np.array(value, dtype=getattr(np, self.config.dtype)))
        return value

    def _snapshot_recurrent_state(self) -> Dict[str, Any]:
        """Capture minimal mutable state needed to make forward() non-mutating.

        Only saves recurrent state (outputs, activations, p matrices, gate
        state, ln_std).  Does NOT save optimizer or LR scheduler state —
        those are unaffected by a bare forward() call.
        """
        state: Dict[str, Any] = {
            'outputs':      self._clone_scalar_or_tensor(self.outputs),
            'activations':  self._clone_scalar_or_tensor(self.activations),
            'errors':       self._clone_scalar_or_tensor(self.errors),
            'p_matrix':     self._clone_scalar_or_tensor(self.p_matrix),
            'p_matrix_old': self._clone_scalar_or_tensor(self.p_matrix_old),
        }
        if self.config.gated:
            state['gate_values']  = self._clone_scalar_or_tensor(self.gate_values)
            state['prev_outputs'] = self._clone_scalar_or_tensor(self.prev_outputs)
            state['candidate']    = self._clone_scalar_or_tensor(
                getattr(self, 'candidate', None))
        if self.config.layer_norm:
            state['ln_std'] = self._clone_scalar_or_tensor(self.ln_std)
        return state

    def _restore_recurrent_state(self, state: Dict[str, Any]) -> None:
        """Restore recurrent state saved by _snapshot_recurrent_state."""
        for key, value in state.items():
            if value is not None:
                setattr(self, key, self._restore_state_value(value))

    # ------------------------------------------------------------------
    # Activation functions and derivatives
    # ------------------------------------------------------------------

    def _activate(self, x, kind: str):
        x = self.B.clamp(x, -self.config.max_activation, self.config.max_activation)
        if kind == "sigmoid":
            return self.B.sigmoid(x)
        elif kind == "tanh":
            return self.B.tanh(x)
        elif kind == "relu":
            return self.B.relu(x)
        elif kind == "leaky_relu":
            return self.B.leaky_relu(x)
        elif kind == "linear":
            return x
        raise ValueError(f"Unknown activation: {kind}")

    def _derivative(self, act, out, kind: str):
        if kind == "sigmoid":
            d = out * (1.0 - out)
        elif kind == "tanh":
            d = 1.0 - out * out
        elif kind == "relu":
            if self.B.use_torch:
                d = (act > 0).to(self.B.dtype)
            else:
                d = (act > 0).astype(self.B.dtype)
        elif kind == "leaky_relu":
            if self.B.use_torch:
                d = torch.where(act > 0, torch.ones_like(act),
                                torch.full_like(act, 0.01))
            else:
                d = np.where(act > 0, 1.0, 0.01).astype(self.B.dtype)
        elif kind == "linear":
            if self.B.use_torch:
                return torch.ones_like(act)
            else:
                return np.ones_like(act)
        else:
            raise ValueError(f"Unknown activation: {kind}")
        return self.B.clamp(d, mn=self.config.y_prime_min)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs, teacher_outputs=None):
        """Forward pass with optional layer norm and gating. Returns z."""
        B = self.B
        z = B.zeros(self.ncols)
        z[0] = 1.0
        z[1:self.m] = inputs

        if teacher_outputs is not None and self.config.teacher_forcing:
            for g in range(self.num_groups):
                gs = g * self.group_size
                z[self.m + gs] = teacher_outputs[g]
                z[self.m + gs + 1:self.m + gs + self.group_size] = \
                    self.outputs[gs + 1:gs + self.group_size]
        else:
            z[self.m:] = self.outputs

        # Save previous outputs for gating
        if self.config.gated:
            self.prev_outputs = B.clone(self.outputs)

        # s = W @ z
        self.activations = B.mv(self.weights, z)

        # Layer normalization (pre-activation)
        if self.config.layer_norm:
            self.activations, self.ln_std = self._layer_norm(self.activations)

        # Compute candidate activation
        candidate = self._activate(self.activations, self.config.hidden_activation)
        if self.config.output_activation != self.config.hidden_activation:
            oi = self.output_indices
            candidate[oi] = self._activate(self.activations[oi],
                                           self.config.output_activation)

        # Apply GRU-style update gate
        if self.config.gated:
            self.candidate = self.B.clone(candidate)  # save for gate gradient
            gate_act = B.mv(self.gate_weights, z) + self.gate_bias
            self.gate_values = B.sigmoid(gate_act)
            # gate≈1 → keep old output; gate≈0 → use new candidate
            self.outputs = self.gate_values * self.prev_outputs + \
                           (1.0 - self.gate_values) * candidate
        else:
            self.outputs = candidate

        return z

    def _layer_norm(self, s):
        """Pre-activation layer normalization. Returns (normed, std)."""
        B = self.B
        n = self.nrows
        if B.use_torch:
            mean = s.mean()
            var = s.var(unbiased=False)
            std = (var + self.config.ln_eps).sqrt()
            s_norm = (s - mean) / std
            return self.ln_gamma * s_norm + self.ln_beta, std
        else:
            mean = s.mean()
            var = s.var()
            std = np.sqrt(var + self.config.ln_eps)
            s_norm = (s - mean) / std
            return self.ln_gamma * s_norm + self.ln_beta, std

    # ------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------

    def compute_error(self, desired) -> float:
        oi = self.output_indices
        output_vals = self.outputs[oi]
        self.errors = desired - output_vals
        return float(0.5 * self.B.sum(self.errors ** 2))

    # ------------------------------------------------------------------
    # Weight update (subgrouped RTRL core)
    # ------------------------------------------------------------------

    def update_weights(self, z):
        """Subgrouped RTRL weight update with optimizer and gradient clipping.

        Steps:
            1. Compute raw gradient from P matrix into grad_buffer
            2. Clip gradient norm (if configured)
            3. Apply optimizer step (Adam or SGD)
            4. Update P matrix for next timestep
            5. Swap P matrices
        """
        B = self.B

        # --- 1. Compute raw gradient ---
        self.grad_buffer[:] = 0
        for g in range(self.num_groups):
            gs = g * self.group_size
            ge = gs + self.group_size
            p_slice = self.p_matrix_old[:, :, gs]           # [S, C]
            self.grad_buffer[gs:ge] += float(self.errors[g]) * p_slice

        # --- 2. Gradient clipping ---
        if self.config.grad_clip_norm > 0:
            gnorm = B.norm(self.grad_buffer)
            if gnorm > self.config.grad_clip_norm:
                self.grad_buffer *= (self.config.grad_clip_norm / (gnorm + 1e-8))

        # --- 3. Optimizer step ---
        if self.config.optimizer == "adam":
            self._adam_step()
        else:
            self._sgd_step()

        # --- 3b. Update gate weights (separate from main RTRL) ---
        if self.config.gated:
            self._update_gate_weights(z)

        # --- 4. Update P matrix ---
        self._update_p_matrix(z)

        # --- 5. Swap P matrices ---
        self.p_matrix_old, self.p_matrix = self.p_matrix, self.p_matrix_old

    def _adam_step(self):
        """Apply Adam optimizer using grad_buffer."""
        cfg = self.config
        B = self.B
        self.adam_t += 1
        t = self.adam_t

        # Weight decay (decoupled, AdamW style)
        if cfg.weight_decay > 0:
            self.weights *= (1.0 - self._current_lr * cfg.weight_decay)

        # Update biased moments
        self.adam_m = cfg.adam_beta1 * self.adam_m + (1 - cfg.adam_beta1) * self.grad_buffer
        self.adam_v = cfg.adam_beta2 * self.adam_v + (1 - cfg.adam_beta2) * (self.grad_buffer ** 2)

        # Bias correction
        bc1 = 1.0 - cfg.adam_beta1 ** t
        bc2 = 1.0 - cfg.adam_beta2 ** t
        m_hat = self.adam_m / bc1
        v_hat = self.adam_v / bc2

        # Update weights: w += lr * m_hat / (sqrt(v_hat) + eps)
        # Note: positive because grad_buffer = (desired - output) * P,
        # which points in the direction of improvement (not negated)
        self.weights += self._current_lr * m_hat / (B.sqrt(v_hat) + cfg.adam_eps)

    def _sgd_step(self):
        """Apply SGD with momentum. v = μv + α∇L; θ += v."""
        self.delta_weights = self.config.momentum * self.delta_weights + \
            self._current_lr * self.grad_buffer
        self.weights += self.delta_weights

    def _update_gate_weights(self, z):
        """Update gate weights using Adam with error-derived gradient.

        NOTE: This uses instantaneous (BPTT(0)) gradients for the gate
        parameters, NOT full RTRL Jacobian tracking.  Extending the P-matrix
        to track ∂y/∂W_gate would roughly double memory.  The local-gradient
        approximation is standard practice for gated units and empirically
        sufficient for the gate to learn open/close behavior.

        For neuron k in group g:
            ∂y_k/∂g_k = prev_out_k - candidate_k
            ∂E/∂g_k = -e_g * (prev_out_k - candidate_k)   (for output neuron)
            ∂g_k/∂W_gate[k,j] = g_k * (1-g_k) * z_j

        Hidden neurons use their group's output error as signal.
        """
        B = self.B
        cfg = self.config
        self.gate_adam_t += 1
        t = self.gate_adam_t
        lr = self._current_lr

        # Compute gate gradients for all neurons
        gate_deriv = self.gate_values * (1.0 - self.gate_values)  # σ'(a) = σ(a)(1-σ(a))
        diff = self.prev_outputs - self.candidate                 # [N]

        # Error signal: broadcast group error to all neurons in group
        error_signal = B.zeros(self.nrows)
        for g in range(self.num_groups):
            gs = g * self.group_size
            ge = gs + self.group_size
            error_signal[gs:ge] = float(self.errors[g])

        # ∂E/∂g_k = -error * diff, then chain through sigmoid derivative
        # Gate grad for weights: [N] * [C] outer product → [N, C]
        gate_grad_scalar = error_signal * diff * gate_deriv       # [N]
        # Outer product: gate_grad_W[k,j] = gate_grad_scalar[k] * z[j]
        if B.use_torch:
            gate_grad_W = gate_grad_scalar[:, None] * z[None, :]  # [N, C]
        else:
            gate_grad_W = gate_grad_scalar[:, np.newaxis] * z[np.newaxis, :]
        gate_grad_b = gate_grad_scalar                            # [N]

        # Adam update for gate weights
        b1, b2, eps = cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps

        # Decoupled weight decay (AdamW-style) — prevents gate saturation
        if cfg.weight_decay > 0:
            self.gate_weights *= (1.0 - lr * cfg.weight_decay)

        self.gate_adam_m_w = b1 * self.gate_adam_m_w + (1 - b1) * gate_grad_W
        self.gate_adam_v_w = b2 * self.gate_adam_v_w + (1 - b2) * (gate_grad_W ** 2)
        m_hat_w = self.gate_adam_m_w / (1 - b1 ** t)
        v_hat_w = self.gate_adam_v_w / (1 - b2 ** t)
        self.gate_weights += lr * m_hat_w / (B.sqrt(v_hat_w) + eps)

        self.gate_adam_m_b = b1 * self.gate_adam_m_b + (1 - b1) * gate_grad_b
        self.gate_adam_v_b = b2 * self.gate_adam_v_b + (1 - b2) * (gate_grad_b ** 2)
        m_hat_b = self.gate_adam_m_b / (1 - b1 ** t)
        v_hat_b = self.gate_adam_v_b / (1 - b2 ** t)
        self.gate_bias += lr * m_hat_b / (B.sqrt(v_hat_b) + eps)

    def _update_p_matrix(self, z):
        """Vectorized P matrix with optional gating and layer norm.

        NOTE: The P-matrix uses a block-diagonal approximation — each
        subgroup's Jacobian only tracks recurrent connections within that
        subgroup (W_rec is sliced as weights[gs:ge, m+gs:m+ge]).  Cross-group
        recurrent influences are intentionally excluded.  This is the core
        contribution of Dean (1994) Subgrouped RTRL, reducing complexity
        from O(N^4) to O(N^3/G) at the cost of approximate gradients.

        Without gating:
            P[i,j,k] = f'(k) * (rec_sum[i,j,k] + kron[i,j,k])

        With gating:
            P[i,j,k] = g_k * P_old[i,j,k] + (1-g_k) * f'(k) * (rec_sum + kron)

        With layer norm (diagonal approx):
            f'(k) is scaled by gamma_k / (std + eps)
        """
        oi = self.output_indices
        h_deriv = self._derivative(self.activations, self.outputs,
                                   self.config.hidden_activation)
        o_deriv = self._derivative(self.activations[oi], self.outputs[oi],
                                   self.config.output_activation)
        all_deriv = self.B.clone(h_deriv)
        all_deriv[oi] = o_deriv

        # Layer norm: scale derivatives by gamma / (std + eps)
        if self.config.layer_norm:
            if self.B.use_torch:
                ln_scale = self.ln_gamma / (self.ln_std + self.config.ln_eps)
            else:
                ln_scale = self.ln_gamma / (float(self.ln_std) + self.config.ln_eps)
            all_deriv = all_deriv * ln_scale

        # Kronecker term: δ_{ik} * z_j → [S, C, S]
        if self.B.use_torch:
            kron = self._eye[:, None, :] * z[None, :, None]
        else:
            kron = self._eye[:, np.newaxis, :] * z[np.newaxis, :, np.newaxis]

        for g in range(self.num_groups):
            gs = g * self.group_size
            ge = gs + self.group_size
            W_rec = self.weights[gs:ge, self.m + gs:self.m + ge]
            P_old_g = self.p_matrix_old[:, :, gs:ge]
            rec_sum = self.B.matmul(P_old_g, W_rec.T)

            if self.config.teacher_forcing:
                rec_sum[:, :, 0] = 0.0

            fprime = all_deriv[gs:ge]
            if self.B.use_torch:
                new_contrib = fprime[None, None, :] * (rec_sum + kron)
            else:
                new_contrib = fprime[np.newaxis, np.newaxis, :] * (rec_sum + kron)

            # Apply gating to P matrix
            if self.config.gated:
                gv = self.gate_values[gs:ge]
                if self.B.use_torch:
                    self.p_matrix[:, :, gs:ge] = \
                        gv[None, None, :] * P_old_g + \
                        (1.0 - gv[None, None, :]) * new_contrib
                else:
                    self.p_matrix[:, :, gs:ge] = \
                        gv[np.newaxis, np.newaxis, :] * P_old_g + \
                        (1.0 - gv[np.newaxis, np.newaxis, :]) * new_contrib
            else:
                self.p_matrix[:, :, gs:ge] = new_contrib

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------

    def check_accuracy(self, desired) -> bool:
        oi = self.output_indices
        output_vals = self.outputs[oi]
        if self.config.categorical_output:
            return self.B.argmax(output_vals) == self.B.argmax(desired)
        else:
            return float(self.B.sum((output_vals - desired) ** 2)) <= self.config.ok_threshold

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self):
        if not self.config.continuous_epochs:
            self.outputs[:] = 0
            self.p_matrix_old[:] = 0
        self.activations[:] = 0
        if self.config.optimizer == "sgd":
            self.delta_weights[:] = 0
        if self.config.gated:
            self.prev_outputs[:] = 0
            self.gate_values[:] = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, data, targets) -> Dict[str, float]:
        total_error = 0.0
        correct = 0
        skipped = 0
        n = len(data)

        self.reset_state()

        for t in range(n):
            if t >= self.config.time_delay or self.config.continuous_epochs:
                teacher = targets[max(0, t - 1)] if self.config.teacher_forcing else None
                z = self.forward(data[t], teacher)

                target_idx = t - self.config.time_delay
                if 0 <= target_idx < len(targets):
                    err = self.compute_error(targets[target_idx])
                    total_error += err
                    if self.check_accuracy(targets[target_idx]):
                        correct += 1
                    if err > self.config.skip_threshold:
                        self.update_weights(z)
                    else:
                        skipped += 1
            else:
                self.forward(data[t])

        return {
            'total_error': total_error,
            'avg_error': total_error / n,
            'accuracy': correct / n * 100.0,
            'skip_pct': skipped / n * 100.0,
        }

    def train(self, data, targets) -> List[Dict[str, Any]]:
        """Full training loop with LR scheduling."""
        if isinstance(data, np.ndarray):
            data = self.B.from_numpy(data)
        if isinstance(targets, np.ndarray):
            targets = self.B.from_numpy(targets)

        history = []
        min_error = float('inf')
        cfg = self.config

        for epoch in range(cfg.epochs):
            stats = self.train_epoch(data, targets)
            stats['epoch'] = epoch + 1
            stats['lr'] = self._current_lr
            history.append(stats)

            # LR scheduling
            te = stats['total_error']
            self._step_scheduler(epoch, te)
            min_error = min(min_error, te)

            if cfg.verbose and (epoch % max(1, cfg.epochs // 20) == 0
                                or epoch == cfg.epochs - 1):
                print(f"  Epoch {epoch+1:4d}/{cfg.epochs}  "
                      f"err={stats['avg_error']:.6f}  "
                      f"acc={stats['accuracy']:.1f}%  "
                      f"lr={self._current_lr:.2e}")

            # Early stop if LR bottomed out (thesis mode)
            if cfg.lr_schedule == "thesis" and self._current_lr < cfg.lr_min:
                if cfg.verbose:
                    print(f"  LR below min, stopping at epoch {epoch+1}")
                break

        self.training_history = history
        return history

    def _step_scheduler(self, epoch: int, total_error: float):
        """Update learning rate based on schedule."""
        sched = self.config.lr_schedule

        if sched == "fixed":
            pass  # no change

        elif sched == "thesis":
            # Original aggressive decay: α *= 0.1 on error increase or stall
            cfg = self.config
            if not hasattr(self, '_thesis_min_error'):
                self._thesis_min_error = float('inf')
            if total_error > self._thesis_min_error * cfg.lr_decay_threshold or \
               abs(total_error - self._thesis_min_error) < 1e-7:
                self._current_lr *= cfg.lr_decay_factor
                if cfg.verbose:
                    print(f"  LR → {self._current_lr:.2e}")
            self._thesis_min_error = min(self._thesis_min_error, total_error)

        elif sched == "plateau":
            # Reduce on plateau with patience
            cfg = self.config
            if total_error < self._plateau_best - 1e-6:
                self._plateau_best = total_error
                self._plateau_wait = 0
            else:
                self._plateau_wait += 1
                if self._plateau_wait >= cfg.plateau_patience:
                    new_lr = max(self._current_lr * cfg.plateau_factor,
                                 cfg.plateau_min_lr)
                    if new_lr < self._current_lr:
                        self._current_lr = new_lr
                        if cfg.verbose:
                            print(f"  Plateau: LR → {self._current_lr:.2e}")
                    self._plateau_wait = 0

        elif sched == "cosine":
            # Cosine annealing
            cfg = self.config
            T = cfg.cosine_T_max if cfg.cosine_T_max > 0 else cfg.epochs
            eta_min = cfg.cosine_eta_min
            self._current_lr = eta_min + (self._base_lr - eta_min) * \
                (1 + math.cos(math.pi * (epoch + 1) / T)) / 2

        else:
            raise ValueError(f"Unknown lr_schedule: {sched}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, data) -> np.ndarray:
        if isinstance(data, np.ndarray):
            data = self.B.from_numpy(data)
        saved = self.B.clone(self.outputs)
        self.reset_state()
        preds = []
        for t in range(len(data)):
            self.forward(data[t])
            preds.append(self.B.to_numpy(self.outputs[self.output_indices]).copy())
        self.outputs = saved
        return np.array(preds)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        sd = {
            'weights': self.B.to_numpy(self.weights).tolist(),
            'config': self.config.__dict__,
            'training_history': getattr(self, 'training_history', []),
        }
        if self.config.gated:
            sd['gate_weights'] = self.B.to_numpy(self.gate_weights).tolist()
            sd['gate_bias'] = self.B.to_numpy(self.gate_bias).tolist()
        if self.config.layer_norm:
            sd['ln_gamma'] = self.B.to_numpy(self.ln_gamma).tolist()
            sd['ln_beta'] = self.B.to_numpy(self.ln_beta).tolist()
        return sd

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.state_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ModernSubgroupedRTRL':
        with open(path) as f:
            d = json.load(f)
        cfg = RTRLConfig(**d['config'])
        net = cls(cfg)
        w = np.array(d['weights'], dtype=getattr(np, cfg.dtype))
        net.weights = net.B.from_numpy(w)
        if cfg.gated and 'gate_weights' in d:
            net.gate_weights = net.B.from_numpy(
                np.array(d['gate_weights'], dtype=getattr(np, cfg.dtype)))
            net.gate_bias = net.B.from_numpy(
                np.array(d['gate_bias'], dtype=getattr(np, cfg.dtype)))
        if cfg.layer_norm and 'ln_gamma' in d:
            net.ln_gamma = net.B.from_numpy(
                np.array(d['ln_gamma'], dtype=getattr(np, cfg.dtype)))
            net.ln_beta = net.B.from_numpy(
                np.array(d['ln_beta'], dtype=getattr(np, cfg.dtype)))
        net.training_history = d.get('training_history', [])
        return net


def create_thesis_config(**kwargs) -> RTRLConfig:
    """Create config matching original thesis defaults."""
    defaults = dict(
        hidden_activation="sigmoid", output_activation="sigmoid",
        init="thesis", per_step_dw_reset=True,
        optimizer="sgd", lr_schedule="thesis",
        lr_decay_factor=0.1, lr_decay_threshold=1.01, y_prime_min=0.01,
    )
    defaults.update(kwargs)
    return RTRLConfig(**defaults)


# ---------------------------------------------------------------------------
# Stage 4: TITANS-Compatible Memory Interface
# ---------------------------------------------------------------------------

@dataclass
class TITANSConfig:
    """Configuration for TITANS surprise-gated memory.

    The RTRL network acts as associative memory: keys (inputs) are mapped
    to values (outputs) via online-learned weights.  Surprise (prediction
    error) gates how aggressively the memory updates — novel information
    gets stored, predictable information is ignored.
    """
    # Dimensions
    key_dim: int                        # Size of input key vectors
    value_dim: int                      # Size of output value vectors
    hidden_dim: int = 32                # Hidden neurons in RTRL network

    # Surprise gating
    surprise_threshold: float = 0.0     # Minimum surprise to trigger write
    surprise_ema_alpha: float = 0.05    # EMA smoothing — lower = slower/more stable baseline
    surprise_modulated_lr: bool = True  # Scale LR by normalized surprise

    # Memory dynamics
    lr: float = 0.003                   # Base learning rate
    weight_decay: float = 1e-5          # Continuous forgetting rate
    momentum_window: int = 0            # Context momentum: accumulate over
                                        # N steps before applying (0=every step)

    # RTRL backbone
    optimizer: str = "adam"
    grad_clip_norm: float = 1.0
    gated: bool = True                  # GRU gate for hidden state retention
    layer_norm: bool = False
    hidden_activation: str = "tanh"
    output_activation: str = "linear"   # Linear for embedding regression
    init: str = "xavier"

    # Runtime
    device: str = "auto"
    dtype: str = "float32"
    verbose: bool = False


class TITANSMemory:
    """TITANS-compatible surprise-gated associative memory.

    Uses a subgrouped RTRL network as a differentiable memory module
    that learns online during inference.  Designed for integration with
    LLMs where:
        key   = query or hidden state embedding
        value = associated information embedding

    Core operations:
        read(query)        → retrieve value without updating memory
        write(key, value)  → update memory (surprise-gated)
        step(key, value)   → combined read-then-write with stats
        surprise(key, val) → compute surprise without updating
        forget(decay)      → apply explicit weight decay

    Surprise mechanics:
        1. Present key to network, get predicted value
        2. Surprise = MSE(predicted, actual_value)
        3. If surprise > threshold → update weights (learn this)
        4. LR optionally scaled by surprise / running_avg_surprise
        5. Momentum buffer can accumulate over N steps before applying

    The GRU gate in the RTRL backbone controls memory retention:
        gate ≈ 1 → retain old memory (familiar context)
        gate ≈ 0 → overwrite with new information (novel context)
    """

    def __init__(self, config: Optional[TITANSConfig] = None, **kwargs):
        """Initialize TITANS memory.

        Args:
            config: TITANSConfig, or pass key_dim/value_dim as kwargs.
        """
        if config is None:
            config = TITANSConfig(**kwargs)
        self.config = config

        # Build RTRL backbone
        rtrl_cfg = RTRLConfig(
            num_inputs=config.key_dim,
            num_outputs=config.value_dim,
            num_hidden=config.hidden_dim,
            time_delay=0,           # No delay for memory read/write
            epochs=1,
            lr=config.lr,
            optimizer=config.optimizer,
            lr_schedule="fixed",
            grad_clip_norm=config.grad_clip_norm,
            weight_decay=config.weight_decay,
            gated=config.gated,
            layer_norm=config.layer_norm,
            hidden_activation=config.hidden_activation,
            output_activation=config.output_activation,
            init=config.init,
            per_step_dw_reset=False,
            continuous_epochs=True,     # Never reset state between calls
            teacher_forcing=False,
            categorical_output=False,
            skip_threshold=0.0,
            y_prime_min=0.01,
            device=config.device,
            dtype=config.dtype,
            verbose=config.verbose,
        )
        self.net = ModernSubgroupedRTRL(rtrl_cfg)
        self.B = self.net.B

        # Surprise tracking
        self._surprise_ema = 0.0        # Exponential moving average of surprise
        self._surprise_count = 0
        self._step_count = 0

        # Momentum buffer (accumulate gradients over N steps)
        self._momentum_steps = 0
        if config.momentum_window > 0:
            self._grad_accum = self.B.zeros_like(self.net.weights)

        # Statistics
        self.stats = {
            'total_writes': 0,
            'total_skipped': 0,
            'total_surprise': 0.0,
            'max_surprise': 0.0,
        }

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def read(self, query) -> np.ndarray:
        """Retrieve value for a query key without mutating memory state.

        Snapshots and restores all recurrent state so that a read() call
        has no effect on subsequent step() or write() calls.

        Args:
            query: [key_dim] numpy array or backend tensor

        Returns:
            value: [value_dim] numpy array
        """
        if isinstance(query, np.ndarray):
            query = self.B.from_numpy(query)
        saved = self.net._snapshot_recurrent_state()
        try:
            self.net.forward(query)
            oi = self.net.output_indices
            return self.B.to_numpy(self.net.outputs[oi]).copy()
        finally:
            self.net._restore_recurrent_state(saved)

    def surprise(self, key, value) -> float:
        """Compute surprise (MSE) for a key-value pair without mutating state.

        Snapshots and restores all recurrent state so that a surprise() call
        has no effect on subsequent step() or write() calls.

        Args:
            key: [key_dim] input vector
            value: [value_dim] target vector

        Returns:
            surprise: scalar MSE prediction error
        """
        if isinstance(key, np.ndarray):
            key = self.B.from_numpy(key)
        if isinstance(value, np.ndarray):
            value = self.B.from_numpy(value)

        saved = self.net._snapshot_recurrent_state()
        try:
            self.net.forward(key)
            oi = self.net.output_indices
            predicted = self.net.outputs[oi]
            diff = value - predicted
            return float(self.B.sum(diff ** 2)) / len(self.net.output_indices)
        finally:
            self.net._restore_recurrent_state(saved)

    def write(self, key, value) -> Dict[str, float]:
        """Update memory with a key-value pair, gated by surprise.

        Args:
            key: [key_dim] input vector
            value: [value_dim] target vector

        Returns:
            dict with 'surprise', 'wrote' (bool), 'effective_lr'
        """
        if isinstance(key, np.ndarray):
            key = self.B.from_numpy(key)
        if isinstance(value, np.ndarray):
            value = self.B.from_numpy(value)

        cfg = self.config

        # Forward pass
        z = self.net.forward(key)

        # Compute surprise (MSE)
        oi = self.net.output_indices
        predicted = self.net.outputs[oi]
        diff = value - predicted
        mse = float(self.B.sum(diff ** 2)) / len(oi)

        # Update surprise EMA
        self._surprise_count += 1
        if self._surprise_count == 1:
            self._surprise_ema = mse
        else:
            alpha = cfg.surprise_ema_alpha
            self._surprise_ema = alpha * mse + (1 - alpha) * self._surprise_ema

        # Track stats
        self.stats['total_surprise'] += mse
        self.stats['max_surprise'] = max(self.stats['max_surprise'], mse)
        self._step_count += 1

        # Decide whether to write
        wrote = mse > cfg.surprise_threshold
        effective_lr = cfg.lr

        if wrote:
            # Compute error signal for RTRL
            self.net.errors[:] = diff  # Note: diff = value - predicted

            # Modulate LR by normalized surprise
            if cfg.surprise_modulated_lr and self._surprise_ema > 1e-10:
                surprise_ratio = min(mse / self._surprise_ema, 5.0)
                effective_lr = cfg.lr * surprise_ratio
                saved_lr = self.net._current_lr
                self.net._current_lr = effective_lr

            # Apply weight update via RTRL
            if cfg.momentum_window > 0:
                self._accumulate_and_maybe_apply(z)
            else:
                self.net.update_weights(z)

            # Restore LR
            if cfg.surprise_modulated_lr and self._surprise_ema > 1e-10:
                self.net._current_lr = saved_lr

            self.stats['total_writes'] += 1
        else:
            # Still update P matrix for gradient tracking even if we skip write
            self.net._update_p_matrix(z)
            self.net.p_matrix_old, self.net.p_matrix = \
                self.net.p_matrix, self.net.p_matrix_old
            self.stats['total_skipped'] += 1

        return {
            'surprise': mse,
            'wrote': wrote,
            'effective_lr': effective_lr,
            'surprise_ema': self._surprise_ema,
        }

    def step(self, key, value) -> Dict[str, Any]:
        """Combined read-then-write: retrieve current prediction, then update.

        Uses a single forward pass — snapshots prediction before write.

        Args:
            key: [key_dim] input vector
            value: [value_dim] target vector

        Returns:
            dict with 'predicted' (before update), 'surprise', 'wrote', etc.
        """
        if isinstance(key, np.ndarray):
            key = self.B.from_numpy(key)
        if isinstance(value, np.ndarray):
            value = self.B.from_numpy(value)

        cfg = self.config

        # Single forward pass
        z = self.net.forward(key)
        oi = self.net.output_indices
        predicted = self.B.to_numpy(self.net.outputs[oi]).copy()

        # Compute surprise (MSE)
        diff = value - self.net.outputs[oi]
        mse = float(self.B.sum(diff ** 2)) / len(oi)

        # Update surprise EMA
        self._surprise_count += 1
        if self._surprise_count == 1:
            self._surprise_ema = mse
        else:
            alpha = cfg.surprise_ema_alpha
            self._surprise_ema = alpha * mse + (1 - alpha) * self._surprise_ema

        self.stats['total_surprise'] += mse
        self.stats['max_surprise'] = max(self.stats['max_surprise'], mse)
        self._step_count += 1

        # Decide whether to write
        wrote = mse > cfg.surprise_threshold
        effective_lr = cfg.lr

        if wrote:
            self.net.errors[:] = diff

            if cfg.surprise_modulated_lr and self._surprise_ema > 1e-10:
                surprise_ratio = min(mse / self._surprise_ema, 5.0)
                effective_lr = cfg.lr * surprise_ratio
                saved_lr = self.net._current_lr
                self.net._current_lr = effective_lr

            if cfg.momentum_window > 0:
                self._accumulate_and_maybe_apply(z)
            else:
                self.net.update_weights(z)

            if cfg.surprise_modulated_lr and self._surprise_ema > 1e-10:
                self.net._current_lr = saved_lr

            self.stats['total_writes'] += 1
        else:
            self.net._update_p_matrix(z)
            self.net.p_matrix_old, self.net.p_matrix = \
                self.net.p_matrix, self.net.p_matrix_old
            self.stats['total_skipped'] += 1

        return {
            'predicted': predicted,
            'surprise': mse,
            'wrote': wrote,
            'effective_lr': effective_lr,
            'surprise_ema': self._surprise_ema,
        }

    # ------------------------------------------------------------------
    # Momentum buffer
    # ------------------------------------------------------------------

    def _accumulate_and_maybe_apply(self, z):
        """Accumulate gradients over momentum_window steps, then apply."""
        net = self.net

        # Compute gradient into net.grad_buffer
        net.grad_buffer[:] = 0
        for g in range(net.num_groups):
            gs = g * net.group_size
            ge = gs + net.group_size
            p_slice = net.p_matrix_old[:, :, gs]
            net.grad_buffer[gs:ge] += float(net.errors[g]) * p_slice

        # Clip
        if net.config.grad_clip_norm > 0:
            gnorm = net.B.norm(net.grad_buffer)
            if gnorm > net.config.grad_clip_norm:
                net.grad_buffer *= (net.config.grad_clip_norm / (gnorm + 1e-8))

        # Accumulate
        self._grad_accum += net.grad_buffer
        self._momentum_steps += 1

        if self._momentum_steps >= self.config.momentum_window:
            # Average and apply
            net.grad_buffer = self._grad_accum / self._momentum_steps
            if net.config.optimizer == "adam":
                net._adam_step()
            else:
                net._sgd_step()
            if net.config.gated:
                net._update_gate_weights(z)
            # Reset accumulator
            self._grad_accum[:] = 0
            self._momentum_steps = 0

        # Always update P matrix
        net._update_p_matrix(z)
        net.p_matrix_old, net.p_matrix = net.p_matrix, net.p_matrix_old

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def forget(self, decay: Optional[float] = None):
        """Apply explicit weight decay (forgetting).

        Args:
            decay: Override weight_decay rate (default: config.weight_decay)
        """
        rate = decay if decay is not None else self.config.weight_decay
        if rate > 0:
            self.net.weights *= (1.0 - rate)

    def reset(self):
        """Reset memory state (clear hidden state and P matrix, keep weights)."""
        self.net.outputs[:] = 0
        self.net.activations[:] = 0
        self.net.p_matrix[:] = 0
        self.net.p_matrix_old[:] = 0
        if self.config.gated:
            self.net.prev_outputs[:] = 0
            self.net.gate_values[:] = 0

    def reset_full(self):
        """Full reset including weights (new memory)."""
        self.net._init_weights()
        self.net._init_state()
        self.net._init_optimizer()
        self._surprise_ema = 0.0
        self._surprise_count = 0
        self._step_count = 0
        self.stats = {k: 0 if isinstance(v, int) else 0.0
                      for k, v in self.stats.items()}

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def write_ratio(self) -> float:
        """Fraction of steps that resulted in memory writes."""
        total = self.stats['total_writes'] + self.stats['total_skipped']
        return self.stats['total_writes'] / max(1, total)

    @property
    def avg_surprise(self) -> float:
        return self.stats['total_surprise'] / max(1, self._step_count)

    def summary(self) -> str:
        return (f"TITANSMemory: {self.config.key_dim}→{self.config.value_dim} "
                f"(hidden={self.config.hidden_dim}, "
                f"gated={self.config.gated}) "
                f"writes={self.stats['total_writes']} "
                f"skipped={self.stats['total_skipped']} "
                f"write_ratio={self.write_ratio:.1%} "
                f"avg_surprise={self.avg_surprise:.4f} "
                f"surprise_ema={self._surprise_ema:.4f}")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str, include_hidden_state: bool = True):
        """Save memory to JSON.

        Args:
            path: File path
            include_hidden_state: If True, saves outputs, P matrix, and gate
                state so a resumed session continues exactly. If False, only
                saves learned weights (faster restart, re-warms in ~few tokens).
        """
        sd = self.net.state_dict()
        sd['titans_config'] = self.config.__dict__
        sd['titans_stats'] = self.stats
        sd['surprise_ema'] = self._surprise_ema
        sd['surprise_count'] = self._surprise_count
        sd['step_count'] = self._step_count
        if include_hidden_state:
            sd['hidden_state'] = {
                'outputs': self.B.to_numpy(self.net.outputs).tolist(),
                'activations': self.B.to_numpy(self.net.activations).tolist(),
                'p_matrix_old': self.B.to_numpy(self.net.p_matrix_old).tolist(),
            }
            if self.config.gated:
                sd['hidden_state']['prev_outputs'] = \
                    self.B.to_numpy(self.net.prev_outputs).tolist()
                sd['hidden_state']['gate_values'] = \
                    self.B.to_numpy(self.net.gate_values).tolist()
        with open(path, 'w') as f:
            json.dump(sd, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TITANSMemory':
        with open(path) as f:
            d = json.load(f)
        tcfg = TITANSConfig(**d['titans_config'])
        dt = getattr(np, tcfg.dtype)
        mem = cls(tcfg)
        B = mem.B

        # Restore learned weights
        mem.net.weights = B.from_numpy(np.array(d['weights'], dtype=dt))
        if tcfg.gated and 'gate_weights' in d:
            mem.net.gate_weights = B.from_numpy(
                np.array(d['gate_weights'], dtype=dt))
            mem.net.gate_bias = B.from_numpy(
                np.array(d['gate_bias'], dtype=dt))

        # Restore tracking state
        mem.stats = d.get('titans_stats', mem.stats)
        mem._surprise_ema = d.get('surprise_ema', 0.0)
        mem._surprise_count = d.get('surprise_count', 0)
        mem._step_count = d.get('step_count', 0)

        # Restore hidden state (if saved)
        hs = d.get('hidden_state')
        if hs:
            mem.net.outputs = B.from_numpy(np.array(hs['outputs'], dtype=dt))
            mem.net.activations = B.from_numpy(
                np.array(hs['activations'], dtype=dt))
            mem.net.p_matrix_old = B.from_numpy(
                np.array(hs['p_matrix_old'], dtype=dt))
            if tcfg.gated and 'prev_outputs' in hs:
                mem.net.prev_outputs = B.from_numpy(
                    np.array(hs['prev_outputs'], dtype=dt))
                mem.net.gate_values = B.from_numpy(
                    np.array(hs['gate_values'], dtype=dt))

        return mem


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("Stage 4 Self-Test: TITANS Memory Interface")
    print("=" * 65)

    results = []

    # -- Test 1: Basic read/write --
    print("\n--- Test 1: Basic read/write associative memory ---")
    mem = TITANSMemory(key_dim=4, value_dim=2, hidden_dim=8,
                       lr=0.01, gated=True, verbose=False)
    # Train on 3 key-value pairs, repeated
    keys = np.eye(4, dtype=np.float32)[:3]
    vals = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)

    for epoch in range(50):
        for k, v in zip(keys, vals):
            mem.write(k, v)

    # Read back
    print(f"  After 50 epochs of 3 patterns:")
    total_err = 0
    for k, v in zip(keys, vals):
        pred = mem.read(k)
        err = np.mean((pred - v) ** 2)
        total_err += err
        print(f"    key={k} → pred={pred.round(3)}  tgt={v}  mse={err:.4f}")
    avg_err = total_err / 3
    print(f"  Avg MSE: {avg_err:.4f}")
    results.append(("1. Basic read/write", avg_err, None))

    # -- Test 2: Surprise gating --
    print("\n--- Test 2: Surprise gating ---")
    mem2 = TITANSMemory(key_dim=4, value_dim=2, hidden_dim=8,
                        lr=0.01, surprise_threshold=0.01,
                        gated=True, verbose=False)
    # Write same pattern many times — surprise should drop, writes should stop
    k_rep = np.array([1, 0, 0, 0], dtype=np.float32)
    v_rep = np.array([1, 0], dtype=np.float32)
    surprises = []
    for i in range(100):
        result = mem2.write(k_rep, v_rep)
        surprises.append(result['surprise'])

    print(f"  Surprise: first={surprises[0]:.4f} → last={surprises[-1]:.6f}")
    print(f"  {mem2.summary()}")
    results.append(("2. Surprise gating",
                     surprises[-1], mem2.write_ratio))

    # -- Test 3: Surprise-modulated LR --
    print("\n--- Test 3: Surprise-modulated LR ---")
    mem3 = TITANSMemory(key_dim=4, value_dim=2, hidden_dim=8,
                        lr=0.005, surprise_modulated_lr=True,
                        surprise_threshold=0.0, gated=True, verbose=False)
    # Alternate between familiar and novel patterns
    k_familiar = np.array([1, 0, 0, 0], dtype=np.float32)
    v_familiar = np.array([1, 0], dtype=np.float32)
    k_novel = np.array([0, 0, 0, 1], dtype=np.float32)
    v_novel = np.array([0, 1], dtype=np.float32)

    # Pre-train on familiar
    for _ in range(30):
        mem3.write(k_familiar, v_familiar)

    # Now present novel — should get high effective LR
    r_novel = mem3.write(k_novel, v_novel)
    # Present familiar again — should get low effective LR
    r_familiar = mem3.write(k_familiar, v_familiar)
    print(f"  Novel:    surprise={r_novel['surprise']:.4f}  "
          f"eff_lr={r_novel['effective_lr']:.6f}")
    print(f"  Familiar: surprise={r_familiar['surprise']:.4f}  "
          f"eff_lr={r_familiar['effective_lr']:.6f}")
    print(f"  LR ratio (novel/familiar): "
          f"{r_novel['effective_lr']/max(r_familiar['effective_lr'], 1e-10):.1f}x")
    results.append(("3. Surprise-modulated LR",
                     r_novel['effective_lr'], r_familiar['effective_lr']))

    # -- Test 4: Momentum window --
    print("\n--- Test 4: Momentum buffer (accumulate over 5 steps) ---")
    mem4 = TITANSMemory(key_dim=4, value_dim=2, hidden_dim=8,
                        lr=0.01, momentum_window=5,
                        gated=True, verbose=False)
    for epoch in range(50):
        for k, v in zip(keys, vals):
            mem4.write(k, v)
    total_err4 = 0
    for k, v in zip(keys, vals):
        pred = mem4.read(k)
        total_err4 += np.mean((pred - v) ** 2)
    avg_err4 = total_err4 / 3
    print(f"  Avg MSE: {avg_err4:.4f}  (cf. Test 1: {avg_err:.4f})")
    results.append(("4. Momentum buffer", avg_err4, None))

    # -- Test 5: step() combined interface --
    print("\n--- Test 5: step() read-then-write ---")
    mem5 = TITANSMemory(key_dim=4, value_dim=2, hidden_dim=8,
                        lr=0.01, gated=True, verbose=False)
    first_step = mem5.step(keys[0], vals[0])
    print(f"  First step: pred={first_step['predicted'].round(3)}  "
          f"surprise={first_step['surprise']:.4f}")
    for epoch in range(30):
        for k, v in zip(keys, vals):
            mem5.step(k, v)
    last_step = mem5.step(keys[0], vals[0])
    print(f"  After 30 epochs: pred={last_step['predicted'].round(3)}  "
          f"surprise={last_step['surprise']:.6f}")
    results.append(("5. step() interface", last_step['surprise'], None))

    # -- Test 6: Forgetting --
    print("\n--- Test 6: Explicit forgetting ---")
    mem6 = TITANSMemory(key_dim=4, value_dim=2, hidden_dim=8,
                        lr=0.01, gated=True, verbose=False)
    for epoch in range(50):
        for k, v in zip(keys, vals):
            mem6.write(k, v)
    pre_forget = mem6.read(keys[0])
    mem6.forget(decay=0.5)  # Aggressive 50% weight decay
    post_forget = mem6.read(keys[0])
    print(f"  Before forget: pred={pre_forget.round(3)}")
    print(f"  After  forget: pred={post_forget.round(3)}")
    results.append(("6. Forgetting", None, None))

    # -- Test 7: Save/load round-trip with hidden state --
    print("\n--- Test 7: Save/load with hidden state persistence ---")
    mem7 = TITANSMemory(key_dim=4, value_dim=2, hidden_dim=8,
                        lr=0.01, gated=True, verbose=False)
    for epoch in range(20):
        for k, v in zip(keys, vals):
            mem7.write(k, v)
    # Save WITH hidden state (before any reads)
    mem7.save("/tmp/titans_hs.json", include_hidden_state=True)
    mem7_loaded = TITANSMemory.load("/tmp/titans_hs.json")
    # Read same key from both — same hidden state → same result
    pred_orig = mem7.read(keys[0])
    pred_loaded = mem7_loaded.read(keys[0])
    diff_with_hs = np.max(np.abs(pred_orig - pred_loaded))
    # Save WITHOUT hidden state
    mem7.save("/tmp/titans_cold.json", include_hidden_state=False)
    mem7_cold = TITANSMemory.load("/tmp/titans_cold.json")
    pred_cold = mem7_cold.read(keys[0])
    diff_cold = np.max(np.abs(pred_orig - pred_cold))
    print(f"  With hidden state:    diff={diff_with_hs:.2e}  "
          f"{'PASS' if diff_with_hs < 1e-5 else 'FAIL'}")
    print(f"  Without (cold start): diff={diff_cold:.2e}  "
          f"(expected non-zero, re-warms quickly)")

    # -- Test 8: Larger memory for embedding-like vectors --
    print("\n--- Test 8: 64-dim embedding memory (LLM-like) ---")
    np.random.seed(42)
    mem8 = TITANSMemory(key_dim=64, value_dim=64, hidden_dim=32,
                        lr=0.003, surprise_threshold=0.001,
                        surprise_modulated_lr=True,
                        grad_clip_norm=1.0, gated=True, verbose=False)
    # Generate 20 random key-value pairs (simulating LLM embeddings)
    kv_keys = np.random.randn(20, 64).astype(np.float32)
    kv_keys /= np.linalg.norm(kv_keys, axis=1, keepdims=True)
    kv_vals = np.random.randn(20, 64).astype(np.float32) * 0.1

    t0 = time.time()
    for epoch in range(20):
        for k, v in zip(kv_keys, kv_vals):
            mem8.write(k, v)
    elapsed = time.time() - t0

    # Recall accuracy
    total_mse = 0
    for k, v in zip(kv_keys, kv_vals):
        pred = mem8.read(k)
        total_mse += np.mean((pred - v) ** 2)
    avg_mse8 = total_mse / 20
    print(f"  20 patterns, 20 epochs: avg_mse={avg_mse8:.6f}  time={elapsed:.2f}s")
    print(f"  {mem8.summary()}")
    results.append(("8. 64-dim embeddings", avg_mse8, None))

    # -- Test 9: RTRL backbone regression (Stage 3 still works) --
    print("\n--- Test 9: RTRL backbone regression ---")
    np.random.seed(42)
    N2 = 1000
    seq_in, seq_tgt = [], []
    last_a = False
    for _ in range(N2):
        c = np.random.randint(0, 4)
        inp = np.zeros(4, dtype=np.float32)
        inp[c] = 1.0
        seq_tgt.append([1.0 if (c == 1 and last_a) else 0.0])
        last_a = (c == 0)
        seq_in.append(inp)
    seq_in, seq_tgt = np.array(seq_in), np.array(seq_tgt)

    cfg9 = RTRLConfig(
        num_inputs=4, num_outputs=1, num_hidden=2,
        time_delay=1, epochs=100, lr=0.003,
        optimizer="adam", lr_schedule="fixed",
        gated=True, hidden_activation="tanh", output_activation="sigmoid",
        init="xavier", categorical_output=False, verbose=False)
    net9 = ModernSubgroupedRTRL(cfg9)
    h9 = net9.train(seq_in, seq_tgt)
    print(f"  Internal State: err={h9[-1]['avg_error']:.6f}  "
          f"acc={h9[-1]['accuracy']:.1f}%")
    results.append(("9. RTRL regression (gated)", h9[-1]['avg_error'],
                     h9[-1]['accuracy']))

    # -- Summary --
    print("\n" + "=" * 65)
    print("Summary:")
    for label, v1, v2 in results:
        extra = ""
        if v2 is not None:
            extra = f"  ({v2})"
        val = f"{v1:.6f}" if v1 is not None else "—"
        print(f"  {label:40s} {val}{extra}")

    print("\n" + "=" * 65)
    print("All 4 stages complete.")
    print("  Stage 1: Core RTRL, Xavier init, vectorized P matrix")
    print("  Stage 2: Adam optimizer, gradient clipping, LR scheduling")
    print("  Stage 3: GRU gating, layer normalization")
    print("  Stage 4: TITANS surprise-gated memory interface")
    print("=" * 65)
