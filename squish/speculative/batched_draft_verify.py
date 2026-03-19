"""batched_draft_verify.py — Cross-Request Batched Speculative Verification

When N concurrent requests are each generating with speculative decoding,
the standard approach runs N independent verification forward passes:

    for req in requests:
        logits = model(req.context + req.draft_tokens, req.kv_cache)

This is N × the Metal dispatch overhead even when each request is small.

BatchedDraftVerifier groups all pending verification calls into a single
batched forward pass:

    batch = [req.draft_tokens for req in requests]
    padded = pad_sequences(batch, pad_id=0)         # (N, max_draft_len)
    batch_logits = model(padded, batch_kv, masks)   # single Metal dispatch
    for req, logits in zip(requests, split(batch_logits)):
        req.accepted = accept(req.draft_tokens, logits)

Cost: 1 forward pass amortised over N requests vs N separate passes.
Ideal batch occupancy: 4-16 concurrent spec-decode requests.

The verifier does NOT handle the actual model forward — it manages request
grouping, padding, un-padding, and per-request acceptance logic.  The caller
provides a ``model_forward_fn`` callback.

Token acceptance rule (standard spec decode):
  For each draft position i: accept if argmax(logits[i]) == draft_tokens[i].
  On first mismatch, accept the target token at position i and drop the rest.

Verification callback signature:
    def model_forward_fn(
        token_ids: np.ndarray,   # (batch, seq_len)
        kv_cache_ids: list[str], # per-request KV cache identifier
        attention_mask: np.ndarray,  # (batch, seq_len)  0=pad
    ) -> np.ndarray:             # (batch, seq_len, vocab_size) logits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class VerifyRequest:
    """A single pending verification request.

    Args:
        request_id:    Unique identifier for this request
        draft_tokens:  Speculative draft token IDs (variable length).
        context_ids:   Token IDs in the context (before draft).
        kv_cache_id:   Identifier for the KV cache entry to use.
    """
    request_id: str
    draft_tokens: list[int]
    context_ids: list[int]
    kv_cache_id: str


@dataclass
class VerifyResult:
    """Verification outcome for a single request.

    Args:
        request_id:      Request identifier (matches VerifyRequest).
        accepted_tokens: Accepted tokens (prefix of the draft that matched).
        bonus_token:     Target-model token at the first mismatch position
                         (or at len(draft) if all accepted).
        n_accepted:      Number of draft tokens accepted.
        n_drafted:       Number of draft tokens proposed.
    """
    request_id: str
    accepted_tokens: list[int]
    bonus_token: int
    n_accepted: int
    n_drafted: int


@dataclass
class BatchedDraftVerifierConfig:
    """Configuration for BatchedDraftVerifier.

    Args:
        max_draft_len:    Maximum number of speculative draft tokens.
                          Longer drafts are truncated.
        pad_id:           Token ID used for sequence padding.
        acceptance_mode:  'greedy' — accept if argmax matches draft.
                          'sample' — sample from residual distribution
                          (not yet implemented, reserved).
        min_batch_size:   Minimum requests to batch before dispatching.
                          If pending < min_batch_size, dispatch anyway if
                          max_wait_ms has elapsed.
    """
    max_draft_len: int = 8
    pad_id: int = 0
    acceptance_mode: str = "greedy"  # 'greedy' | 'sample'
    min_batch_size: int = 1


@dataclass
class BatchVerifyStats:
    """Runtime statistics tracked by BatchedDraftVerifier."""
    total_batches: int = 0
    total_requests: int = 0
    total_drafted: int = 0
    total_accepted: int = 0

    @property
    def accept_rate(self) -> float:
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    @property
    def mean_requests_per_batch(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return self.total_requests / self.total_batches


# Model-forward callback type alias
ModelForwardFn = Callable[
    [np.ndarray, list[str], np.ndarray],  # (token_ids, kv_ids, mask)
    np.ndarray,                            # (batch, seq_len, vocab)
]


class BatchedDraftVerifier:
    """Groups N concurrent spec-decode verification calls into one forward pass.

    Lifecycle:
        1. One or more ``add_request()`` calls to populate the pending queue.
        2. ``verify_all(model_forward_fn)`` to dispatch the batched forward pass
           and return per-request VerifyResult objects.
        3. Repeat from step 1.

    The verifier is stateless between calls to verify_all() — it drains the
    pending queue on each call.
    """

    def __init__(self, config: Optional[BatchedDraftVerifierConfig] = None) -> None:
        self.config = config or BatchedDraftVerifierConfig()
        self._pending: list[VerifyRequest] = []
        self.stats = BatchVerifyStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, request: VerifyRequest) -> None:
        """Enqueue a verification request.

        Args:
            request: VerifyRequest with draft tokens and context.
        """
        if not request.request_id:
            raise ValueError("request_id must be non-empty")
        if not request.draft_tokens:
            raise ValueError("draft_tokens must be non-empty")

        cfg = self.config
        # Truncate draft to max_draft_len
        if len(request.draft_tokens) > cfg.max_draft_len:
            request = VerifyRequest(
                request_id=request.request_id,
                draft_tokens=request.draft_tokens[: cfg.max_draft_len],
                context_ids=request.context_ids,
                kv_cache_id=request.kv_cache_id,
            )
        self._pending.append(request)

    def pending_count(self) -> int:
        """Number of requests currently waiting to be verified."""
        return len(self._pending)

    def verify_all(
        self,
        model_forward_fn: ModelForwardFn,
    ) -> dict[str, VerifyResult]:
        """Run one batched verification pass over all pending requests.

        Calls model_forward_fn exactly once with the padded batch.

        Returns:
            Mapping from request_id → VerifyResult.

        Raises:
            RuntimeError: If there are no pending requests.
        """
        if not self._pending:
            raise RuntimeError("No pending requests to verify")

        requests = list(self._pending)
        self._pending.clear()

        cfg = self.config
        n = len(requests)

        # Find the longest draft in this batch
        max_len = max(len(r.draft_tokens) for r in requests)

        # Build padded token_ids and attention_mask (batch, max_draft_len)
        token_ids = np.full((n, max_len), fill_value=cfg.pad_id, dtype=np.int32)
        mask = np.zeros((n, max_len), dtype=np.bool_)
        for i, req in enumerate(requests):
            dlen = len(req.draft_tokens)
            token_ids[i, :dlen] = req.draft_tokens
            mask[i, :dlen] = True

        kv_ids = [r.kv_cache_id for r in requests]

        # Single batched forward pass — (batch, max_len, vocab_size)
        batch_logits = model_forward_fn(token_ids, kv_ids, mask.astype(np.int32))

        if batch_logits.shape[:2] != (n, max_len):
            raise ValueError(
                f"model_forward_fn returned shape {batch_logits.shape}, "
                f"expected ({n}, {max_len}, vocab)"
            )

        # Per-request acceptance
        results: dict[str, VerifyResult] = {}
        for i, req in enumerate(requests):
            draft = req.draft_tokens
            n_drafted = len(draft)
            logits_i = batch_logits[i, :n_drafted, :]  # (n_drafted, vocab)

            accepted: list[int] = []
            bonus = int(np.argmax(logits_i[-1]))

            if cfg.acceptance_mode == "greedy":
                for t_idx, token_id in enumerate(draft):
                    greedy_token = int(np.argmax(logits_i[t_idx]))
                    if greedy_token == token_id:
                        accepted.append(token_id)
                    else:
                        # Mismatch — accept the target token instead
                        bonus = greedy_token
                        break

            n_accepted = len(accepted)
            # Bonus token is target prediction at first mismatch (or end+1)
            results[req.request_id] = VerifyResult(
                request_id=req.request_id,
                accepted_tokens=accepted,
                bonus_token=bonus,
                n_accepted=n_accepted,
                n_drafted=n_drafted,
            )

            self.stats.total_drafted += n_drafted
            self.stats.total_accepted += n_accepted

        self.stats.total_batches += 1
        self.stats.total_requests += n
        return results

    def reset_stats(self) -> None:
        self.stats = BatchVerifyStats()

    def __repr__(self) -> str:
        return (
            f"BatchedDraftVerifier(pending={len(self._pending)}, "
            f"batches={self.stats.total_batches}, "
            f"accept_rate={self.stats.accept_rate:.2%})"
        )
