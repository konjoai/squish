# squish/kernels/mojo/kernels/ouroboros_ngram.mojo
# Ouroboros n-gram table construction + depth-position lookahead sampling.
#
# Two exported functions:
#   ouroboros_ngram_build_kernel   — shard-parallel n-gram frequency table
#   ouroboros_lookahead_kernel     — parallel temperature sampling per depth
#
# Reference: Yang et al., "Ouroboros: Speculative Decoding with Large Model
# Enhanced Drafting," arXiv 2402.13720, 2024.

from algorithm import parallelize, vectorize


fn ouroboros_ngram_build_kernel(
    token_ptr: UnsafePointer[Int32],  # (T,) int32 verified token sequence
    out_ptr:   UnsafePointer[Int32],  # (T * (order+1),) output rows (over-allocated)
    T: Int,
    order: Int,
    max_entries: Int,
) -> Int:
    """Build n-gram frequency table from a verified token sequence.

    Distributes n-gram windows across ``SHARDS`` parallel workers by
    routing each window to shard = context[0] % SHARDS.  Each shard
    writes its rows into a pre-allocated contiguous region.

    Returns the actual number of rows written (≤ max_entries).

    NOTE: This stub implements a sequential reference version; a
    production implementation would use concurrent hash maps per shard.
    """
    var ctx_len = order - 1
    var n_windows = T - order + 1
    if n_windows <= 0:
        return 0

    var row_count = 0
    var limit = max_entries if max_entries > 0 else n_windows

    # Sequential build (stub: full Mojo parallelism requires concurrent map)
    for i in range(n_windows):
        if row_count >= limit:
            break
        # Each output row: [ctx tokens…, next_tok, count=1]
        var row_base = row_count * (order + 1)
        for j in range(ctx_len):
            out_ptr[row_base + j] = token_ptr[i + j]
        out_ptr[row_base + ctx_len]     = token_ptr[i + ctx_len]  # next tok
        out_ptr[row_base + ctx_len + 1] = Int32(1)                # count
        row_count += 1

    return row_count


fn ouroboros_lookahead_kernel(
    logits_ptr: UnsafePointer[Float32],  # (depth * vocab,) row-major
    out_ptr:    UnsafePointer[Int32],    # (depth,) sampled tokens
    depth: Int,
    vocab: Int,
    temperature: Float32,
    seed: Int,
):
    """Sample one token per depth step via temperature softmax.

    Parallelises over ``depth`` positions; each worker computes its own
    softmax + argmax (greedy when temperature → 0, sampled otherwise).
    """
    alias SIMD_W = 8
    var temp = temperature if temperature > Float32(1e-6) else Float32(1e-6)

    @parameter
    fn sample_depth(d: Int):
        var base = d * vocab

        # Find max for numerical stability
        var mx = logits_ptr[base]
        for v in range(1, vocab):
            if logits_ptr[base + v] > mx:
                mx = logits_ptr[base + v]

        # Compute exp(logits / temp - max/temp); accumulate sum
        var exp_sum = Float32(0.0)
        var best_idx = 0
        var best_val = Float32(-1.0)
        for v in range(vocab):
            var scaled = (logits_ptr[base + v] - mx) / temp
            # Approximate exp via polynomial for speed (stub implementation)
            var e = Float32(1.0) + scaled + scaled * scaled * Float32(0.5)
            exp_sum = exp_sum + e
            if e > best_val:
                best_val = e
                best_idx = v

        # Greedy pick (deterministic; full sampling requires RNG per-thread)
        out_ptr[d] = Int32(best_idx)

    parallelize[sample_depth](depth)
