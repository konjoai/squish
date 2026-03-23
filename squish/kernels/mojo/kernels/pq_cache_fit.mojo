# squish/kernels/mojo/kernels/pq_cache_fit.mojo
# PQ sub-codebook centroid fitting via Lloyd's algorithm.
#
# Outer Lloyd loop is sequential; inner E-step assignment parallelises over
# the N sub-vectors.  M-step centroid update is a sequential scatter-reduce.
#
# Reference: Jégou et al., "Product Quantization for Nearest Neighbor Search,"
# IEEE TPAMI 2011.

from algorithm import parallelize


fn pq_cache_fit_kernel(
    sub_vecs_ptr:   UnsafePointer[Float32],  # (n * sub_dim) flat
    centroids_ptr:  UnsafePointer[Float32],  # (k * sub_dim) output
    n:              Int,
    sub_dim:        Int,
    k:              Int,
    n_iters:        Int,
    seed:           Int,
):
    """Lloyd clustering for PQ sub-codebook fitting.

    Initialises centroids from evenly-spaced sub-vectors, then runs
    ``n_iters`` Lloyd iterations with parallel E-step over N vectors and
    sequential M-step centroid scatter-reduce.

    Args:
        sub_vecs_ptr:  Flat float32 sub-vector matrix (n × sub_dim).
        centroids_ptr: Output centroid matrix (k × sub_dim).
        n:             Number of sub-vectors.
        sub_dim:       Sub-vector dimension.
        k:             Number of centroids.
        n_iters:       Lloyd iterations.
        seed:          Seed (unused; deterministic init from spacing).
    """
    # Initialise centroids from evenly-spaced samples
    for ci in range(k):
        var idx = (ci * n // k) if k <= n else ci % n
        for d in range(sub_dim):
            centroids_ptr[ci * sub_dim + d] = sub_vecs_ptr[idx * sub_dim + d]

    var assignments = UnsafePointer[Int32].alloc(n)

    for _iter in range(n_iters):
        # E-step: parallel assignment
        @parameter
        fn assign_vec(i: Int):
            var best_dist = Float32(1e38)
            var best_ci = 0
            for ci in range(k):
                var dist = Float32(0.0)
                for d in range(sub_dim):
                    var diff = sub_vecs_ptr[i * sub_dim + d] - centroids_ptr[ci * sub_dim + d]
                    dist += diff * diff
                if dist < best_dist:
                    best_dist = dist
                    best_ci = ci
            assignments[i] = Int32(best_ci)

        parallelize[assign_vec](n)

        # M-step: sequential centroid scatter-reduce
        for ci in range(k):
            var sum_buf = UnsafePointer[Float64].alloc(sub_dim)
            for d in range(sub_dim):
                sum_buf[d] = Float64(0.0)
            var cnt = Int64(0)
            for i in range(n):
                if Int(assignments[i]) == ci:
                    for d in range(sub_dim):
                        sum_buf[d] += Float64(sub_vecs_ptr[i * sub_dim + d])
                    cnt += 1
            if cnt > 0:
                for d in range(sub_dim):
                    centroids_ptr[ci * sub_dim + d] = Float32(sum_buf[d] / Float64(cnt))
            sum_buf.free()

    assignments.free()
