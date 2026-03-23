# squish/kernels/mojo/kernels/jacobi_convergence.mojo
# Jacobi fixed-point convergence check.
#
# Generates updated token guesses for N speculative positions in parallel:
#   temperature == 0 → argmax over vocabulary
#   temperature > 0  → Gumbel-max sampling (xorshift64 per position)
# Then counts positions where guess is unchanged (converged).
# Parallelises over N positions; inner loop vectorised over vocab.
#
# Reference: Santilli et al., "Accelerating Transformer Inference for
# Translation via Parallel Decoding," ACL 2023.

from algorithm import parallelize, vectorize
import math


fn jacobi_convergence_kernel(
    logits_ptr: UnsafePointer[Float32],   # (n * vocab,)  FP32 per-position logits
    guesses_ptr: UnsafePointer[Int32],    # (n,)          Int32 previous guesses
    new_guess_ptr: UnsafePointer[Int32],  # (n,)          Int32 output new guesses
    n_fixed_ptr: UnsafePointer[Int32],    # (1,)          Int32 convergence count
    n: Int,
    vocab: Int,
    temperature: Float32,
    seed: Int64,
):
    """Generate next Jacobi iteration guesses for N speculative positions."""
    var total_fixed = Int32(0)

    @parameter
    fn check_pos(pos: Int):
        var off = pos * vocab

        var new_g: Int32
        if temperature <= Float32(0.0):
            # Greedy argmax
            var best_val = Float32(-1.0e30)
            var best_idx = Int32(0)
            for v in range(vocab):
                var lv = logits_ptr[off + v]
                if lv > best_val:
                    best_val = lv
                    best_idx = Int32(v)
            new_g = best_idx
        else:
            # Gumbel-max sampling (deterministic xorshift64 per position)
            var state = UInt64(
                Int64(seed)
                + Int64(pos) * Int64(6364136223846793005)
                + Int64(1442695040888963407)
            )
            var max_val = Float32(-1.0e30)
            for v in range(vocab):
                state ^= state << 13
                state ^= state >> 7
                state ^= state << 17
                var u = Float32(state >> 33) / Float32(4294967295) + Float32(1e-20)
                var gumbel = logits_ptr[off + v] / temperature - math.log(-math.log(u))
                if gumbel > max_val:
                    max_val = gumbel
                    new_g = Int32(v)

        new_guess_ptr[pos] = new_g
        if new_g == guesses_ptr[pos]:
            total_fixed += Int32(1)

    parallelize[check_pos](n)
    n_fixed_ptr[0] = total_fixed
