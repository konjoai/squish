# squish/kernels/mojo/kernels/cluster_kv_score.mojo
# ClusterKV: per-cluster attention weight aggregation.
#
# Accumulates attention weights into cluster score buckets.
# Parallelised over cluster shards; atomic-free by sharding the
# token sequence into blocks and reducing per-shard.
#
# Reference: Wu et al., "ClusterKV: Manipulating LLM KV Cache in
# Semantic Space for Recallable Compression," arXiv 2412.03213, 2024.

from algorithm import parallelize, vectorize


fn cluster_kv_score_kernel(
    assign_ptr: UnsafePointer[Int32],    # (S,) cluster index per token
    attn_ptr:   UnsafePointer[Float32],  # (S,) attention weight per token
    out_ptr:    UnsafePointer[Float32],  # (n_clusters,) output scores (zeroed)
    S: Int,
    n_clusters: Int,
):
    """Aggregate token attention weights into per-cluster scores.

    Parallelises by dividing tokens into ``n_clusters`` shards; each
    shard accumulates its own slice and writes directly to ``out_ptr``.
    The sharding by cluster id makes writes non-overlapping when
    ``n_clusters`` is a reasonable partition count.
    """
    alias SHARD_SIZE = 256

    var n_shards = (S + SHARD_SIZE - 1) // SHARD_SIZE

    @parameter
    fn process_shard(shard: Int):
        var start = shard * SHARD_SIZE
        var end   = start + SHARD_SIZE
        if end > S:
            end = S
        for i in range(start, end):
            var c = Int(assign_ptr[i])
            if c >= 0 and c < n_clusters:
                out_ptr[c] = out_ptr[c] + attn_ptr[i]

    parallelize[process_shard](n_shards)
