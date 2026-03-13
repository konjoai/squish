"""tests/test_scheduler_bucket_argus_unit.py — 100% coverage for BucketServe/Argus additions in squish/scheduler.py"""
import math

import numpy as np
import pytest

from squish.serving.scheduler import (
    BucketBounds,
    OutputLengthPredictor,
    assign_bucket,
    build_default_buckets,
)

# ---------------------------------------------------------------------------
# BucketBounds
# ---------------------------------------------------------------------------

class TestBucketBounds:
    def test_valid_construction(self):
        b = BucketBounds(64, 127, "s")
        assert b.min_tokens == 64
        assert b.max_tokens == 127
        assert b.label      == "s"

    def test_default_label_empty(self):
        b = BucketBounds(0, 100)
        assert b.label == ""

    def test_invalid_min_tokens(self):
        with pytest.raises(ValueError, match="min_tokens"):
            BucketBounds(-1, 100)

    def test_invalid_max_less_than_min(self):
        with pytest.raises(ValueError, match="max_tokens"):
            BucketBounds(50, 10)

    def test_equal_min_max_ok(self):
        b = BucketBounds(64, 64, "exact")
        assert b.contains(64)

    def test_contains_within(self):
        b = BucketBounds(64, 127)
        assert b.contains(64)
        assert b.contains(100)
        assert b.contains(127)

    def test_contains_outside(self):
        b = BucketBounds(64, 127)
        assert not b.contains(63)
        assert not b.contains(128)


# ---------------------------------------------------------------------------
# build_default_buckets
# ---------------------------------------------------------------------------

class TestBuildDefaultBuckets:
    def test_returns_six_buckets(self):
        buckets = build_default_buckets()
        assert len(buckets) == 6

    def test_labels(self):
        labels = [b.label for b in build_default_buckets()]
        assert labels == ["xs", "s", "m", "l", "xl", "xxl"]

    def test_xs_range(self):
        b = build_default_buckets()[0]
        assert b.min_tokens == 0
        assert b.max_tokens == 63

    def test_xxl_range(self):
        b = build_default_buckets()[-1]
        assert b.min_tokens == 1024
        assert b.max_tokens == 4095

    def test_contiguous(self):
        buckets = build_default_buckets()
        for i in range(len(buckets) - 1):
            assert buckets[i].max_tokens + 1 == buckets[i + 1].min_tokens


# ---------------------------------------------------------------------------
# assign_bucket
# ---------------------------------------------------------------------------

class TestAssignBucket:
    def test_assign_xs(self):
        b = assign_bucket(10)
        assert b.label == "xs"

    def test_assign_s(self):
        b = assign_bucket(64)
        assert b.label == "s"

    def test_assign_m(self):
        b = assign_bucket(200)
        assert b.label == "m"

    def test_assign_l(self):
        b = assign_bucket(300)
        assert b.label == "l"

    def test_assign_xl(self):
        b = assign_bucket(512)
        assert b.label == "xl"

    def test_assign_xxl(self):
        b = assign_bucket(2000)
        assert b.label == "xxl"

    def test_out_of_range_falls_back_to_last(self):
        b = assign_bucket(99999)
        assert b == build_default_buckets()[-1]

    def test_custom_buckets(self):
        custom = [BucketBounds(0, 9, "tiny"), BucketBounds(10, 99, "big")]
        assert assign_bucket(5, custom).label  == "tiny"
        assert assign_bucket(50, custom).label == "big"

    def test_custom_buckets_fallback(self):
        custom = [BucketBounds(0, 9, "tiny")]
        # 100 doesn't fit → last bucket
        b = assign_bucket(100, custom)
        assert b.label == "tiny"


# ---------------------------------------------------------------------------
# OutputLengthPredictor
# ---------------------------------------------------------------------------

class TestOutputLengthPredictor:
    def test_defaults(self):
        p = OutputLengthPredictor()
        assert p.n_samples == 0
        assert p.weights.shape == (4,)

    def test_invalid_default_length(self):
        with pytest.raises(ValueError, match="default_output_length"):
            OutputLengthPredictor(default_output_length=0)

    def test_predict_returns_int(self):
        p = OutputLengthPredictor(default_output_length=100)
        result = p.predict("Hello how are you?")
        assert isinstance(result, int)
        assert result >= 1

    def test_predict_minimum_one(self):
        """Prediction should never be < 1 even if weights are negative."""
        p = OutputLengthPredictor(default_output_length=1)
        p._weights[:] = -1000.0   # force very negative prediction
        assert p.predict("short") >= 1

    def test_n_samples_increments(self):
        p = OutputLengthPredictor()
        p.update("Write a poem.", 50)
        assert p.n_samples == 1
        p.update("Summarize this.", 100)
        assert p.n_samples == 2

    def test_update_changes_weights(self):
        p    = OutputLengthPredictor(default_output_length=256)
        orig = p.weights.copy()
        p.update("Explain quantum computing.", 500)
        assert not np.allclose(p.weights, orig)

    def test_weights_property_is_copy(self):
        p    = OutputLengthPredictor()
        w    = p.weights
        w[:] = 999.0
        assert p.weights[0] != 999.0   # original not mutated

    def test_detect_task_summarize(self):
        assert OutputLengthPredictor._detect_task("Please summarize this doc.") == 0

    def test_detect_task_compare(self):
        assert OutputLengthPredictor._detect_task("Compare A and B.") == 1

    def test_detect_task_explain(self):
        assert OutputLengthPredictor._detect_task("Explain why") == 2

    def test_detect_task_translate(self):
        assert OutputLengthPredictor._detect_task("Translate to French") == 3

    def test_detect_task_code(self):
        assert OutputLengthPredictor._detect_task("Write some code for me") == 4

    def test_detect_task_generate(self):
        assert OutputLengthPredictor._detect_task("Generate a story") == 5

    def test_detect_task_list(self):
        assert OutputLengthPredictor._detect_task("List the ingredients") == 6

    def test_detect_task_question(self):
        assert OutputLengthPredictor._detect_task("What is the answer?") == 7

    def test_detect_task_other(self):
        assert OutputLengthPredictor._detect_task("blah blah blah") == 8

    def test_featurize_shape(self):
        p    = OutputLengthPredictor()
        feat = p._featurize("hello world this is a test")
        assert feat.shape == (4,)
        # bias
        assert feat[0] == 1.0
        # prompt_len: 6 words
        assert feat[1] == pytest.approx(6.0)
        # log(6+1) ≈ 1.946
        assert feat[2] == pytest.approx(math.log(7), abs=1e-5)

    def test_online_learning_converges(self):
        """After sufficient updates with consistent data, prediction should improve."""
        p = OutputLengthPredictor(default_output_length=10)
        prompt = "Summarize this long document about machine learning."
        for _ in range(100):
            p.update(prompt, 200)
        pred = p.predict(prompt)
        assert abs(pred - 200) < 100   # should be in the right ballpark
