# squish/kernels/mojo/kernels/medusa_verify.mojo
# Medusa speculative tree verification kernel.
#
# For each draft head, checks the acceptance ratio p_target / p_draft
# against the threshold.  Heads are processed in parallel; sequential
# acceptance stops at the first rejected head.
#
# Reference: Cai et al., "Medusa: Simple LLM Inference Acceleration
# Framework with Multiple Decoding Heads." ICML 2024.

from algorithm import parallelize


fn medusa_verify_kernel(
    draft_tokens_ptr: UnsafePointer[Int32],   # (n_heads,) token ids
    draft_probs_ptr: UnsafePointer[Float32],  # (n_heads,) draft probabilities
    target_probs_ptr: UnsafePointer[Float32], # (n_heads * vocab_size,) target probs
    accept_flags_ptr: UnsafePointer[Int32],   # (n_heads,) output accept flags
    n_heads: Int,
    vocab_size: Int,
    accept_threshold: Float32,
):
    # Phase 1: compute per-head acceptance ratio in parallel
    @parameter
    fn check_head(h: Int):
        var tok = draft_tokens_ptr[h]
        var p_draft = draft_probs_ptr[h]
        var p_target = target_probs_ptr[h * vocab_size + tok]
        var ratio = p_target / (p_draft if p_draft > Float32(1e-9) else Float32(1e-9))
        accept_flags_ptr[h] = 1 if ratio >= accept_threshold else 0

    parallelize[check_head](n_heads)

    # Phase 2: enforce prefix acceptance (sequential stop at first rejection)
    for i in range(1, n_heads):
        if accept_flags_ptr[i - 1] == 0:
            accept_flags_ptr[i] = 0
