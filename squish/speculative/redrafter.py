"""
squish/speculative/redrafter.py

Phase 4 — ReDrafter: Recurrent Draft Head for Speculative Decoding

Architecture (Apple AI Research, arxiv 2411.13355)
─────────────────────────────────────────────────
For each draft step:

  Step 0 — Conditioning on target hidden state:
    x  = input_proj( concat([h_T, embed(prev_tok)]) )  # (gru_hidden,)
    h  = GRU(x, h_0=zeros)
    tok_0 = sample( lm_head(h) )

  Steps 1..k-1 — Autoregressive via RNN state only:
    x  = input_proj( concat([zeros(target_dim), embed(tok_i)]) )
    h  = GRU(x, h_prev)
    tok_{i+1} = sample( lm_head(h) )

The RNN carries all per-request state; no KV cache is required.
Typical acceptance rate vs. EAGLE: 60-75 % (slightly lower than EAGLE-3
because the RNN is simpler, but much cheaper to train from scratch on a
small dataset).

Design
──────
• Pure numpy — no MLX required at import time; fully unit-testable without
  Metal/GPU.
• Accepts mx.array for target_hidden (from HiddenStateCapture) and
  converts to float32 numpy internally.
• Identical external interface to EagleDraftHead:
      draft_k(target_hidden, k, prev_token_id, temperature, top_p, eos_id)
      → (list[int], list[np.ndarray])
• Weights saved/loaded as .npz (keys documented in ReDrafterHead.save).
• init_random() produces a randomly-initialized head suitable for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Lazy import so the module loads in CPU-only test environments
if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as mx

# Reuse the same sampling helpers from the parent speculative module
from squish.speculative.speculative import (
    _greedy,  # noqa: PLC2701
    _sample,  # noqa: PLC2701
    _softmax_np,  # noqa: PLC2701
    _top_p_filter,  # noqa: PLC2701
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ReDrafterConfig:
    """
    Hyperparameters for the ReDrafter GRU draft head.

    Parameters
    ----------
    target_hidden_dim : Dimensionality of the target model's hidden states.
                        Must match ``HiddenStateCapture.last_hidden.shape[-1]``.
    embed_dim         : Token embedding dimension (same as target model).
    hidden_dim        : GRU cell hidden size.  Larger → better quality, slower.
    n_layers          : Number of stacked GRU cells (1 or 2 is typical).
    """
    target_hidden_dim: int = 2048
    embed_dim:         int = 2048
    hidden_dim:        int = 512
    n_layers:          int = 1

    def __post_init__(self) -> None:
        if self.target_hidden_dim < 1:
            raise ValueError("target_hidden_dim must be ≥ 1")
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be ≥ 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be ≥ 1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be ≥ 1")


# ---------------------------------------------------------------------------
# GRU cell (pure numpy)
# ---------------------------------------------------------------------------

class ReDrafterGRU:
    """
    Single-layer GRU cell implemented in pure numpy.

    Standard GRU equations (Cho et al., 2014):
        r  = σ( W_r·x + U_r·h + b_r )          reset gate
        z  = σ( W_z·x + U_z·h + b_z )          update gate
        n  = tanh( W_n·x + r ⊙ (U_n·h) + b_n ) candidate hidden
        h' = (1 − z) ⊙ h  +  z ⊙ n

    Parameters
    ----------
    input_dim  : Dimensionality of x.
    hidden_dim : Dimensionality of h (and h').
    W, U       : If None, weights are zeroed (caller's responsibility to set).
    """

    __slots__ = (
        "input_dim",
        "hidden_dim",
        # Gate weights  [input → gates]: shape (3*hidden, input)
        "W",   # float32 (3*hidden_dim, input_dim)  — columns [W_r | W_z | W_n]
        # Recurrent weights [hidden → gates]: shape (3*hidden, hidden)
        "U",   # float32 (3*hidden_dim, hidden_dim)
        # Biases (3*hidden,)
        "b",   # float32 (3*hidden_dim,)
    )

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        W: np.ndarray | None = None,
        U: np.ndarray | None = None,
        b: np.ndarray | None = None,
    ) -> None:
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.W = W if W is not None else np.zeros((3 * hidden_dim, input_dim),  dtype=np.float32)
        self.U = U if U is not None else np.zeros((3 * hidden_dim, hidden_dim), dtype=np.float32)
        self.b = b if b is not None else np.zeros(3 * hidden_dim,               dtype=np.float32)

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        One GRU step.

        Parameters
        ----------
        x : (input_dim,)  float32
        h : (hidden_dim,) float32

        Returns
        -------
        h' : (hidden_dim,) float32
        """
        hd = self.hidden_dim

        # Pre-compute input and recurrent contributions
        gates_x = self.W @ x + self.b               # (3*hidden,)
        gates_h = self.U @ h                         # (3*hidden,)  [no bias for U]

        r = _sigmoid(gates_x[:hd]  + gates_h[:hd])  # reset gate
        z = _sigmoid(gates_x[hd:2*hd] + gates_h[hd:2*hd])  # update gate
        n = np.tanh( gates_x[2*hd:] + r * gates_h[2*hd:])   # candidate

        return ((np.float32(1.0) - z) * h + z * n).astype(np.float32)

    # ── random init helper ─────────────────────────────────────────────────────

    @classmethod
    def random_init(
        cls,
        input_dim:  int,
        hidden_dim: int,
        rng: np.random.Generator | None = None,
    ) -> ReDrafterGRU:
        """Kaiming-uniform initialisation (for tests / from-scratch training)."""
        rng = rng or np.random.default_rng(0)
        scale_W = np.sqrt(2.0 / input_dim)
        scale_U = np.sqrt(2.0 / hidden_dim)
        W = rng.standard_normal((3 * hidden_dim, input_dim)).astype(np.float32)  * scale_W
        U = rng.standard_normal((3 * hidden_dim, hidden_dim)).astype(np.float32) * scale_U
        b = np.zeros(3 * hidden_dim, dtype=np.float32)
        return cls(input_dim, hidden_dim, W=W, U=U, b=b)


# ---------------------------------------------------------------------------
# Draft head
# ---------------------------------------------------------------------------

class ReDrafterHead:
    """
    ReDrafter GRU-based draft head.

    Produces k draft tokens conditioned on the target model's hidden states,
    then continues autoregressively using only the GRU's own recurrent state.

    Parameters
    ----------
    config       : :class:`ReDrafterConfig`
    gru_layers   : list of :class:`ReDrafterGRU` (len == config.n_layers)
    lm_head_w    : (vocab_size, hidden_dim) — tied to target model's lm_head
    embed_w      : (vocab_size, embed_dim)  — tied to target model's embed_tokens
    input_proj_w : (hidden_dim, target_hidden_dim + embed_dim) — projects the
                   concat([h_T, embed]) vector into GRU input space
    input_proj_b : (hidden_dim,) bias for input_proj, or None
    """

    __slots__ = (
        "config",
        "_gru_layers",
        "_lm_head_w",       # (vocab_size, hidden_dim)
        "_embed_w",         # (vocab_size, embed_dim)
        "_proj_w",          # (hidden_dim, target_hidden_dim + embed_dim)
        "_proj_b",          # (hidden_dim,) or None
        "_h_state",         # list[np.ndarray]  — per-layer GRU hidden states
        "_draft_hiddens",   # list[np.ndarray]  — last-layer GRU hidden per draft step
    )

    def __init__(
        self,
        config:       ReDrafterConfig,
        gru_layers:   list[ReDrafterGRU],
        lm_head_w:    np.ndarray,
        embed_w:      np.ndarray,
        input_proj_w: np.ndarray,
        input_proj_b: np.ndarray | None = None,
    ) -> None:
        assert len(gru_layers) == config.n_layers, (
            f"Expected {config.n_layers} GRU layers, got {len(gru_layers)}"
        )
        self.config      = config
        self._gru_layers = gru_layers
        self._lm_head_w  = lm_head_w.astype(np.float32)
        self._embed_w    = embed_w.astype(np.float32)
        self._proj_w     = input_proj_w.astype(np.float32)
        self._proj_b        = input_proj_b.astype(np.float32) if input_proj_b is not None else None
        self._h_state: list[np.ndarray] = self._zero_state()
        self._draft_hiddens: list[np.ndarray] = []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _zero_state(self) -> list[np.ndarray]:
        return [
            np.zeros(self.config.hidden_dim, dtype=np.float32)
            for _ in range(self.config.n_layers)
        ]

    def _proj_input(self, h_T: np.ndarray, embed: np.ndarray) -> np.ndarray:
        """Concatenate target hidden + embedding and project to GRU input dim."""
        x = np.concatenate([h_T, embed], axis=0)   # (target_dim + embed_dim,)
        out = self._proj_w @ x
        if self._proj_b is not None:
            out = out + self._proj_b
        return out                                  # (hidden_dim,)

    def _gru_forward(self, x: np.ndarray) -> np.ndarray:
        """Run stacked GRU layers on input x, updating internal state."""
        h = x
        for i, cell in enumerate(self._gru_layers):
            self._h_state[i] = cell.step(h, self._h_state[i])
            h = self._h_state[i]
        return h   # (hidden_dim,)

    def _lm_head(self, h: np.ndarray) -> np.ndarray:
        """Project GRU hidden → logits."""
        return self._lm_head_w @ h   # (vocab_size,)

    # ── public API ────────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Clear per-request GRU hidden states (call at start of each request)."""
        self._h_state       = self._zero_state()
        self._draft_hiddens = []

    @property
    def draft_hiddens(self) -> list[np.ndarray]:
        """
        Per-step last-layer GRU hidden states from the most recent ``draft_k``
        call.  One entry per draft token, in order.  Used by :class:`SSDPredictor`.
        """
        return self._draft_hiddens

    def draft_k(
        self,
        target_hidden,          # mx.array (1, T, target_hidden_dim) or np.ndarray
        k: int,
        prev_token_id: int,
        temperature: float,
        top_p: float,
        eos_id: int,
    ) -> tuple[list[int], list[np.ndarray]]:
        """
        Produce up to *k* draft tokens.

        Step 0 is conditioned on the target model's last hidden state (h_T).
        Steps 1..k-1 are purely autoregressive via the GRU's recurrent state.

        Parameters
        ----------
        target_hidden : mx.array (1, T, target_hidden_dim) from HiddenStateCapture,
                        or a numpy array of the same shape (useful in tests).
        k             : Maximum number of draft tokens to propose.
        prev_token_id : The last token accepted (or the last prompt token).
        temperature   : Sampling temperature (0.0 → greedy).
        top_p         : Nucleus sampling threshold.
        eos_id        : Token ID that terminates drafting early.

        Returns
        -------
        draft_ids   : list[int]        — proposed token IDs
        draft_probs : list[np.ndarray] — per-token probability vectors (vocab,)
        """
        self.reset_state()
        self._draft_hiddens = []

        # Convert mx.array → numpy (no-op if already numpy)
        if hasattr(target_hidden, "__array__"):
            h_np = np.array(target_hidden, dtype=np.float32)   # (1, T, target_dim)
        else:
            h_np = np.asarray(target_hidden, dtype=np.float32)

        # Last hidden state of the last token position: (target_hidden_dim,)
        h_T = h_np[0, -1]

        draft_ids:   list[int]        = []
        draft_probs: list[np.ndarray] = []

        # ── Step 0: draft the first token conditioned on target hidden ────────
        embed_prev = self._embed_w[prev_token_id]              # (embed_dim,)
        x0         = self._proj_input(h_T, embed_prev)         # (hidden_dim,)
        h_out      = self._gru_forward(x0)                     # (hidden_dim,)
        self._draft_hiddens.append(h_out.copy())
        logits0    = self._lm_head(h_out)                      # (vocab_size,)

        probs0 = _top_p_filter(_softmax_np(logits0, temperature), top_p)
        tok0   = _sample(probs0)
        draft_ids.append(tok0)
        draft_probs.append(probs0)

        if tok0 == eos_id or k == 1:
            return draft_ids, draft_probs

        # ── Steps 1..k-1: autoregressive GRU continuation ────────────────────
        zeros_h_T  = np.zeros(self.config.target_hidden_dim, dtype=np.float32)
        cur_tok    = tok0
        for _ in range(k - 1):
            embed_cur = self._embed_w[cur_tok]                 # (embed_dim,)
            x_i       = self._proj_input(zeros_h_T, embed_cur) # (hidden_dim,)
            h_out     = self._gru_forward(x_i)                 # (hidden_dim,)
            self._draft_hiddens.append(h_out.copy())
            logits_i  = self._lm_head(h_out)                   # (vocab_size,)

            probs_i = _top_p_filter(_softmax_np(logits_i, temperature), top_p)
            tok_i   = _sample(probs_i)
            draft_ids.append(tok_i)
            draft_probs.append(probs_i)

            if tok_i == eos_id:
                break
            cur_tok = tok_i

        return draft_ids, draft_probs

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save weights to a .npz file.

        Keys
        ────
        config_*     : scalar config fields
        proj_w, proj_b : input projection
        gru{i}_W, gru{i}_U, gru{i}_b : per-layer GRU weights
        (lm_head_w and embed_w are intentionally NOT saved — they are always
        loaded from the target model's checkpoint to avoid duplication.)
        """
        arrays: dict[str, np.ndarray] = {
            "config_target_hidden_dim": np.array(self.config.target_hidden_dim),
            "config_embed_dim":         np.array(self.config.embed_dim),
            "config_hidden_dim":        np.array(self.config.hidden_dim),
            "config_n_layers":          np.array(self.config.n_layers),
            "proj_w": self._proj_w,
        }
        if self._proj_b is not None:
            arrays["proj_b"] = self._proj_b
        for i, cell in enumerate(self._gru_layers):
            arrays[f"gru{i}_W"] = cell.W
            arrays[f"gru{i}_U"] = cell.U
            arrays[f"gru{i}_b"] = cell.b
        np.savez(path, **arrays)

    @classmethod
    def load(
        cls,
        path: str,
        lm_head_w: np.ndarray,
        embed_w:   np.ndarray,
    ) -> ReDrafterHead:
        """
        Load a :class:`ReDrafterHead` from a .npz file and attach
        the target model's shared weights.

        Parameters
        ----------
        path      : Path to the .npz produced by :meth:`save`.
        lm_head_w : (vocab_size, hidden_dim)  from target model.
        embed_w   : (vocab_size, embed_dim)   from target model.
        """
        d = np.load(path)
        cfg = ReDrafterConfig(
            target_hidden_dim = int(d["config_target_hidden_dim"]),
            embed_dim         = int(d["config_embed_dim"]),
            hidden_dim        = int(d["config_hidden_dim"]),
            n_layers          = int(d["config_n_layers"]),
        )
        proj_w = d["proj_w"]
        proj_b = d["proj_b"] if "proj_b" in d else None
        gru_layers = [
            ReDrafterGRU(
                input_dim  = cfg.hidden_dim,
                hidden_dim = cfg.hidden_dim,
                W = d[f"gru{i}_W"],
                U = d[f"gru{i}_U"],
                b = d[f"gru{i}_b"],
            )
            for i in range(cfg.n_layers)
        ]
        return cls(cfg, gru_layers, lm_head_w, embed_w, proj_w, proj_b)

    @classmethod
    def init_random(
        cls,
        vocab_size:        int,
        target_hidden_dim: int,
        embed_dim:         int | None    = None,
        hidden_dim:        int           = 512,
        n_layers:          int           = 1,
        rng:               np.random.Generator | None = None,
    ) -> ReDrafterHead:
        """
        Create a randomly-initialised :class:`ReDrafterHead` for testing
        or bootstrapping training.

        The ``lm_head_w`` and ``embed_w`` are also random here; in production
        they must be replaced with the target model's actual weights via
        :meth:`load`.
        """
        rng      = rng or np.random.default_rng(42)
        ed       = embed_dim or target_hidden_dim
        cfg      = ReDrafterConfig(
            target_hidden_dim = target_hidden_dim,
            embed_dim         = ed,
            hidden_dim        = hidden_dim,
            n_layers          = n_layers,
        )
        input_dim = target_hidden_dim + ed
        proj_scale = np.sqrt(2.0 / input_dim)
        proj_w = rng.standard_normal((hidden_dim, input_dim)).astype(np.float32) * proj_scale
        proj_b = np.zeros(hidden_dim, dtype=np.float32)

        gru_layers = [
            ReDrafterGRU.random_init(hidden_dim, hidden_dim, rng=rng)
            for _ in range(n_layers)
        ]

        lm_head_w = rng.standard_normal((vocab_size, hidden_dim)).astype(np.float32) * 0.02
        embed_w   = rng.standard_normal((vocab_size, ed)).astype(np.float32) * 0.02

        return cls(cfg, gru_layers, lm_head_w, embed_w, proj_w, proj_b)

    @classmethod
    def from_dir(
        cls,
        head_dir:     str,
        target_model,
        verbose: bool = False,
    ) -> ReDrafterHead:
        """
        Load a :class:`ReDrafterHead` from *head_dir*.

        Expects a ``redrafter.npz`` file in the directory.
        ``lm_head`` and ``embed_tokens`` weights are shared with the target
        model (standard weight-tying convention; same as EAGLE).

        Parameters
        ----------
        head_dir     : Path to directory containing ``redrafter.npz``.
        target_model : The target model (mlx_lm); used to extract shared weights.
        """
        import logging
        logger = logging.getLogger(__name__)

        head_dir_p = Path(head_dir).expanduser()
        npz_path   = head_dir_p / "redrafter.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"ReDrafter weights not found: {npz_path}\n"
                "  → Expected: {head_dir_p}/redrafter.npz"
            )
        if verbose:
            logger.info("Loading ReDrafter head from %s", npz_path)

        # Shared weight resolution: prefer target model weights
        lm_head_w = _resolve_weight(target_model, "lm_head.weight")
        embed_w   = _resolve_weight(target_model, "model.embed_tokens.weight")

        if lm_head_w is None:
            raise RuntimeError(
                "Cannot locate lm_head.weight in target model. "
                "Ensure the target model is a standard mlx_lm transformer."
            )
        if embed_w is None:
            raise RuntimeError(
                "Cannot locate model.embed_tokens.weight in target model."
            )

        lm_head_np = np.array(lm_head_w, dtype=np.float32)
        embed_np   = np.array(embed_w,   dtype=np.float32)

        head = cls.load(str(npz_path), lm_head_np, embed_np)
        if verbose:
            logger.info(
                "ReDrafter head loaded (vocab=%d, target_dim=%d, gru_dim=%d, layers=%d)",
                embed_np.shape[0], head.config.target_hidden_dim,
                head.config.hidden_dim, head.config.n_layers,
            )
        return head


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic sigmoid (float32, clips extreme values)."""
    x = np.asarray(x)
    # Clip to float32-safe range: exp(88) ≈ 1.65e38 (near float32 max)
    x_c = np.clip(x, -88.0, 88.0)
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-x_c))).astype(np.float32)


def _resolve_weight(model, dotted_attr: str) -> np.ndarray | None:
    """
    Walk a dotted attribute path on *model* and return the weight as numpy,
    or None if any part of the path is absent.

    Examples
    --------
    _resolve_weight(model, "lm_head.weight")
    _resolve_weight(model, "model.embed_tokens.weight")
    """
    obj = model
    for part in dotted_attr.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    try:
        return np.array(obj, dtype=np.float32)
    except Exception:
        return None
