# layer_skip_predict.mojo — LayerSkip per-layer confidence predictor Mojo kernel.
#
# Wave 59b — MojoLayerSkipPredict
#
# Replaces Python list comprehension:
#     [sigmoid(dot(w[l], features)) for l in range(n_layers)]
#
# parallelize(n_layers) tasks; each layer:
#     vectorize SIMD FMA dot-product over n_features
#     scalar sigmoid 1 / (1 + exp(-dot))
#
# Specialization:
#     @parameter on n_features ∈ {16, 32}
#     parallelize over n_layers
#
# Reference:
#     Elhoushi et al. (ACL 2024, arXiv 2404.16710) — LayerSkip.

from algorithm import parallelize, vectorize
from math import exp
from sys.info import simdwidthof

alias float32 = DType.float32
alias simd_width = simdwidthof[float32]()


fn layer_skip_predict[n_features: Int](
    weights: DTypePointer[float32],   # (n_layers, n_features)
    features: DTypePointer[float32],  # (n_features,)
    scores: DTypePointer[float32],    # (n_layers,) — write output
    n_layers: Int,
):
    """
    Compute per-layer skip confidence: scores[l] = sigmoid(dot(weights[l], features)).

    Parallelizes over n_layers; each layer uses vectorized SIMD FMA for the
    dot-product and a scalar sigmoid for the final activation.
    """
    @parameter
    fn predict_layer(l: Int):
        var w_ptr = weights + l * n_features
        var dot: Float32 = 0.0

        @parameter
        fn dot_step[width: Int](i: Int):
            var w_vec = SIMD[float32, width].load(w_ptr + i)
            var f_vec = SIMD[float32, width].load(features + i)
            dot += (w_vec * f_vec).reduce_add()
        vectorize[dot_step, simd_width](n_features)

        # Sigmoid with numerical stability
        var x = dot
        if x >= 0.0:
            scores[l] = 1.0 / (1.0 + exp(-x))
        else:
            var e = exp(x)
            scores[l] = e / (1.0 + e)

    parallelize[predict_layer](n_layers)
