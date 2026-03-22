"""squish/attention/lasp_parallel.py

LASPLinearAttn — Ring-topology sequence-parallel linear attention.

Shards the input sequence across ``n_workers`` ring-connected processes.
Each worker accumulates a local recurrent state (S_t of shape
``(head_dim, head_dim)``) and passes it to the next worker in the ring.
The total data communicated per ring step is O(head_dim²) rather than
O(n_tokens × head_dim), making this communication-efficient for long
sequences.

This NumPy reference simulates the per-worker forward pass and the ring
state-passing loop in a single process.

Reference
---------
Sun et al., "LASP: Efficient Multi-device Linear Attention Sequence
Parallelism." arXiv:2405.01234, 2024.
"""

from __future__ import annotations

__all__ = ["LASPConfig", "LASPRingState", "LASPLinearAttn"]

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LASPConfig:
    """Configuration for LASPLinearAttn.

    Parameters
    ----------
    d_model:
        Input model dimension.
    n_heads:
        Number of attention heads.
    head_dim:
        Per-head dimension.
    n_workers:
        Simulated number of ring-connected workers (sequence shards).
    seed:
        RNG seed for projection weight initialisation.
    """

    d_model: int = 256
    n_heads: int = 4
    head_dim: int = 64
    n_workers: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError("d_model must be >= 1")
        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if self.head_dim < 1:
            raise ValueError("head_dim must be >= 1")
        if self.n_workers < 1:
            raise ValueError("n_workers must be >= 1")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class LASPRingState:
    """Per-worker ring state for LASPLinearAttn.

    Attributes
    ----------
    recv_state:
        Recurrent state matrix received from the previous ring worker,
        shape ``(n_heads, head_dim, head_dim)``.  Zero-initialised at the
        start of the ring.
    worker_id:
        Index of this worker in the ring (0-based).
    n_steps:
        Number of forward steps this state has been used for.
    """

    recv_state: ndarray
    worker_id: int = 0
    n_steps: int = 0


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class LASPLinearAttn:
    """Linear attention with ring sequence parallelism.

    Uses the ``φ(Q)K^TV`` formulation:

        S_t = S_{t-1} + k_t v_t^T      (outer product accumulation)
        o_t = q_t S_t

    where φ is an element-wise feature map (ELU + 1).

    Parameters
    ----------
    config:
        ``LASPConfig`` instance.
    """

    def __init__(self, config: LASPConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = float(config.d_model) ** -0.5
        # Q, K, V projection weights: (n_heads * head_dim, d_model)
        proj_dim = config.n_heads * config.head_dim
        self._W_Q: ndarray = rng.standard_normal((proj_dim, config.d_model)).astype(np.float32) * scale
        self._W_K: ndarray = rng.standard_normal((proj_dim, config.d_model)).astype(np.float32) * scale
        self._W_V: ndarray = rng.standard_normal((proj_dim, config.d_model)).astype(np.float32) * scale
        self._W_O: ndarray = rng.standard_normal((config.d_model, proj_dim)).astype(np.float32) * scale

    def new_state(self, worker_id: int = 0) -> LASPRingState:
        """Create a zero-initialised ring state for ``worker_id``."""
        return LASPRingState(
            recv_state=np.zeros(
                (self.config.n_heads, self.config.head_dim, self.config.head_dim),
                dtype=np.float32,
            ),
            worker_id=worker_id,
        )

    def forward(
        self, x: ndarray, state: LASPRingState
    ) -> Tuple[ndarray, LASPRingState]:
        """Full-sequence forward (single worker, uses recv_state from ring).

        Parameters
        ----------
        x:
            Input sequence, shape ``(T, d_model)``.
        state:
            ``LASPRingState`` carrying the recurrent state from the previous
            worker in the ring.

        Returns
        -------
        out:
            Output, shape ``(T, d_model)``.
        state:
            Updated state carrying the send_state (to be passed to next worker).
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.config.d_model:
            raise ValueError(f"x must be (T, {self.config.d_model}), got {x.shape}")
        T = x.shape[0]
        n_heads = self.config.n_heads
        hd = self.config.head_dim

        # Project
        Q = (x @ self._W_Q.T).reshape(T, n_heads, hd)   # (T, H, d)
        K = (x @ self._W_K.T).reshape(T, n_heads, hd)
        V = (x @ self._W_V.T).reshape(T, n_heads, hd)

        # Feature map φ: ELU + 1 (positive-definite)
        phi_Q = np.where(Q >= 0, Q + 1, np.exp(Q))
        phi_K = np.where(K >= 0, K + 1, np.exp(K))

        out_heads = np.zeros((T, n_heads, hd), dtype=np.float32)
        S = state.recv_state.copy()  # (H, d, d)

        for t in range(T):
            for h in range(n_heads):
                k_t = phi_K[t, h]  # (d,)
                v_t = V[t, h]      # (d,)
                S[h] += np.outer(k_t, v_t)  # (d, d)

                q_t = phi_Q[t, h]  # (d,)
                out_heads[t, h] = q_t @ S[h]  # (d,)

        # Output projection
        out_flat = out_heads.reshape(T, n_heads * hd)
        out = out_flat @ self._W_O.T  # (T, d_model)

        new_state = LASPRingState(
            recv_state=S,
            worker_id=state.worker_id,
            n_steps=state.n_steps + 1,
        )
        return out, new_state

    def ring_step(
        self, local_x: ndarray, recv_state: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """Compute one ring step for ``local_x``, returning send_state.

        Parameters
        ----------
        local_x:
            Local sequence shard, shape ``(T_local, d_model)``.
        recv_state:
            Recurrent state from previous worker, shape
            ``(n_heads, head_dim, head_dim)``.

        Returns
        -------
        local_out:
            Output for this shard, shape ``(T_local, d_model)``.
        send_state:
            Recurrent state to pass to the next worker,
            shape ``(n_heads, head_dim, head_dim)``.
        """
        tmp_state = LASPRingState(recv_state=recv_state.copy())
        local_out, new_state = self.forward(local_x, tmp_state)
        return local_out, new_state.recv_state
