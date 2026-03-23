# squish/kernels/mojo/kernels/paged_kv_gather.mojo
# Paged KV-cache block gather kernel.
#
# Reconstructs a contiguous (n_valid_tokens, n_heads, head_dim) tensor
# from non-contiguous physical pages given a page table.
# Parallelises over tokens; vectorises the head-dim copy.
#
# Reference: Kwon et al., "PagedAttention." SOSP 2023.

from algorithm import parallelize, vectorize


fn paged_kv_gather_kernel(
    pool_ptr: UnsafePointer[Float32],  # (max_blocks * n_heads * block_size, head_dim) flat
    page_table_ptr: UnsafePointer[Int32], # (n_pages,)
    out_ptr: UnsafePointer[Float32],   # (n_valid_tokens * n_heads * head_dim,) output
    n_heads: Int,
    block_size: Int,
    head_dim: Int,
    n_valid_tokens: Int,
):
    alias SIMD_W = 8  # FP32 SIMD width for head-dim copy

    @parameter
    fn gather_token(tok: Int):
        var page_idx = tok // block_size
        var pos_in_page = tok % block_size
        var phys_page = page_table_ptr[page_idx]
        var stride = n_heads * block_size * head_dim

        for h in range(n_heads):
            var src_off = phys_page * stride + h * block_size * head_dim + pos_in_page * head_dim
            var dst_off = (tok * n_heads + h) * head_dim

            @parameter
            fn copy_elem[simd_w: Int](i: Int):
                out_ptr[dst_off + i] = pool_ptr[src_off + i]

            vectorize[copy_elem, SIMD_W](head_dim)

    parallelize[gather_token](n_valid_tokens)
