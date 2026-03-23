# squish/kernels/mojo/kernels/tree_verify.mojo
# Tree-parallel speculative decoding verifier (rejection sampling).
#
# Verifies B draft branches in parallel using token-level rejection
# sampling: for each position, accepts the draft token with probability
# min(1, p_target/p_draft); on rejection, samples from the residual
# max(0, p_target - p_draft) distribution.  Returns the longest accepted
# sequence across all branches.
# Parallelises over branches; inner per-branch loop is sequential.
#
# Reference: Miao et al., "SpecInfer: Accelerating LLM Serving with
# Tree-based Speculative Inference and Verification," ASPLOS 2024.

from algorithm import parallelize, vectorize
import math


fn tree_verify_kernel(
    draft_tokens_ptr: UnsafePointer[Int32],    # (B * n_draft,)          Int32
    draft_logits_ptr: UnsafePointer[Float32],  # (B * n_draft * vocab,)  Float32
    target_logits_ptr: UnsafePointer[Float32], # (B * n_draft * vocab,)  Float32
    out_tokens_ptr: UnsafePointer[Int32],      # (n_draft,)              Int32 best sequence
    best_len_ptr: UnsafePointer[Int32],        # (1,)                    Int32
    n_branches: Int,
    n_draft: Int,
    vocab: Int,
    temperature: Float32,
    seed: Int64,
):
    """Verify draft branches via rejection sampling; return best accepted."""
    var temp = temperature if temperature > Float32(1e-6) else Float32(1e-6)

    # Per-branch accepted lengths (and their last token sequences)
    # are accumulated into a shared best tracked sequentially after
    # the parallel scan.
    var branch_lens_ptr = out_tokens_ptr  # reuse output buffer temporarily

    @parameter
    fn verify_branch(b: Int):
        var rng = UInt64(
            Int64(seed) + Int64(b) * Int64(2654435761)
        )
        var b_off = b * n_draft * vocab
        var b_tok_off = b * n_draft
        var accepted_len = Int32(0)

        for i in range(n_draft):
            # Softmax for draft and target at this position
            var dl_off = b_off + i * vocab
            var tl_off = b_off + i * vocab
            var token = Int(draft_tokens_ptr[b_tok_off + i])
            if token >= vocab:
                token = vocab - 1

            # Compute softmax probabilities
            var d_max = Float32(-1.0e30)
            var t_max = Float32(-1.0e30)
            for v in range(vocab):
                var dv = draft_logits_ptr[dl_off + v] / temp
                var tv = target_logits_ptr[tl_off + v] / temp
                if dv > d_max: d_max = dv
                if tv > t_max: t_max = tv
            var d_sum = Float32(0.0)
            var t_sum = Float32(0.0)
            for v in range(vocab):
                d_sum += math.exp(draft_logits_ptr[dl_off + v] / temp - d_max)
                t_sum += math.exp(target_logits_ptr[tl_off + v] / temp - t_max)

            var p_d = math.exp(draft_logits_ptr[dl_off + token] / temp - d_max) / (d_sum + Float32(1e-9))
            var p_t = math.exp(target_logits_ptr[tl_off + token] / temp - t_max) / (t_sum + Float32(1e-9))
            var accept_prob = p_t / (p_d + Float32(1e-30))
            if accept_prob > Float32(1.0):
                accept_prob = Float32(1.0)

            # xorshift64 random
            rng ^= rng << 13
            rng ^= rng >> 7
            rng ^= rng << 17
            var u = Float32(rng >> 33) / Float32(4294967295)

            if u < accept_prob:
                accepted_len += Int32(1)
            else:
                # Sample correction from residual distribution
                var res_sum = Float32(0.0)
                for v in range(vocab):
                    var res = math.exp(target_logits_ptr[tl_off + v] / temp - t_max) / (t_sum + Float32(1e-9)) \
                              - math.exp(draft_logits_ptr[dl_off + v] / temp - d_max) / (d_sum + Float32(1e-9))
                    if res > Float32(0.0):
                        res_sum += res
                rng ^= rng << 13
                rng ^= rng >> 7
                rng ^= rng << 17
                var r = Float32(rng >> 33) / Float32(4294967295) * res_sum
                var cum = Float32(0.0)
                for v in range(vocab):
                    var res = math.exp(target_logits_ptr[tl_off + v] / temp - t_max) / (t_sum + Float32(1e-9)) \
                              - math.exp(draft_logits_ptr[dl_off + v] / temp - d_max) / (d_sum + Float32(1e-9))
                    if res > Float32(0.0):
                        cum += res
                        if cum >= r:
                            accepted_len += Int32(1)
                            break
                break  # stop this branch after rejection

        branch_lens_ptr[b] = accepted_len

    parallelize[verify_branch](n_branches)

    # Find best branch (sequential, short)
    var best_len = Int32(0)
    var best_b = 0
    for b in range(n_branches):
        if branch_lens_ptr[b] > best_len:
            best_len = branch_lens_ptr[b]
            best_b = b
    best_len_ptr[0] = best_len
    # Copy best branch tokens into out_tokens output
    for i in range(Int(best_len)):
        out_tokens_ptr[i] = draft_tokens_ptr[best_b * n_draft + i]
