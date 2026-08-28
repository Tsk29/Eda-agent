from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda_agent.guardrails.leakage import compute_auc, detect_leakage
from eda_agent.guardrails.multiple_comparisons import apply_benjamini_hochberg
from eda_agent.guardrails.sentinels import detect_sentinels
from eda_agent.guardrails.subgroup_reversal import detect_subgroup_reversal

# ---------------------------------------------------------------------------
# multiple_comparisons
# ---------------------------------------------------------------------------


def test_bh_correction_matches_hand_calculation():
    # p sorted ascending, m = 5. adjusted_i = min_{k=i..m}(p_k * m / k), taken
    # as a running minimum from the largest index down.
    p_values = [0.005, 0.011, 0.02, 0.04, 0.13]
    expected_corrected = [0.025, 0.0275, 0.033333333, 0.05, 0.13]

    result = apply_benjamini_hochberg(p_values, alpha=0.05)

    assert result.raw_p_values == p_values
    assert result.corrected_p_values == pytest.approx(expected_corrected, rel=1e-6)
    assert result.rejected == [True, True, True, True, False]


def test_bh_correction_empty_input():
    result = apply_benjamini_hochberg([])
    assert result.raw_p_values == []
    assert result.corrected_p_values == []
    assert result.rejected == []


def test_bh_correction_rejects_out_of_range_p_value():
    with pytest.raises(ValueError):
        apply_benjamini_hochberg([0.5, 1.2])


def test_bh_correction_preserves_input_order():
    p_values = [0.2, 0.001, 0.05]
    result = apply_benjamini_hochberg(p_values)
    assert result.raw_p_values == p_values
    # smallest raw p-value should not have a larger corrected p-value
    # than a larger raw p-value at a higher index
    assert result.corrected_p_values[1] <= result.corrected_p_values[0]


# ---------------------------------------------------------------------------
# leakage
# ---------------------------------------------------------------------------


def test_compute_auc_perfect_separation():
    feature = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    target = [0, 0, 0, 1, 1, 1]
    assert compute_auc(feature, target) == pytest.approx(1.0)


def test_compute_auc_requires_binary_target():
    with pytest.raises(ValueError):
        compute_auc([1.0, 2.0, 3.0], [0, 1, 2])


def test_detect_leakage_flags_near_perfect_linear_feature_binary_target():
    rng = np.random.default_rng(0)
    target = np.array([0] * 50 + [1] * 50)
    # feature cleanly separates the two classes -> AUC should be ~1.0
    feature = target * 100.0 + rng.uniform(0, 0.01, size=100)

    result = detect_leakage(feature, target, target_type="binary")

    assert result.is_leaky is True
    assert result.auc == pytest.approx(1.0, abs=1e-6)


def test_detect_leakage_does_not_flag_random_noise_binary_target():
    rng = np.random.default_rng(1)
    target = rng.integers(0, 2, size=200)
    feature = rng.normal(size=200)

    result = detect_leakage(feature, target, target_type="binary")

    assert result.is_leaky is False
    assert 0.3 < result.auc < 0.7


def test_detect_leakage_flags_near_perfect_linear_feature_continuous_target():
    rng = np.random.default_rng(2)
    target = rng.normal(size=200)
    feature = 3.0 * target + rng.normal(scale=1e-4, size=200)

    result = detect_leakage(feature, target, target_type="continuous")

    assert result.is_leaky is True
    assert abs(result.correlation) > 0.98


def test_detect_leakage_does_not_flag_random_noise_continuous_target():
    rng = np.random.default_rng(3)
    target = rng.normal(size=200)
    feature = rng.normal(size=200)

    result = detect_leakage(feature, target, target_type="continuous")

    assert result.is_leaky is False
    assert abs(result.correlation) < 0.3


def test_detect_leakage_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        detect_leakage([1.0, 2.0], [1.0, 2.0, 3.0], target_type="continuous")


# ---------------------------------------------------------------------------
# sentinels
# ---------------------------------------------------------------------------


def test_numeric_sentinel_minus_999_flagged_in_non_negative_column():
    series = pd.Series([0, 1, 2, -999, 4, 5], name="col")
    result = detect_sentinels(series)
    assert result.has_sentinels is True
    assert "-999" in result.sentinel_values_found
    assert result.affected_row_count == 1


def test_numeric_sentinel_minus_1_flagged_even_when_in_range():
    # -1 is not a statistical outlier here (it's within the visible range of
    # the rest of the column), but the spec's rule is literal: -1 counts as
    # a sentinel whenever the rest of the column is non-negative.
    series = pd.Series([0, 1, 2, 3, -1, 5, 6], name="col")
    result = detect_sentinels(series)
    assert result.has_sentinels is True
    assert "-1" in result.sentinel_values_found
    assert result.affected_row_count == 1


def test_numeric_sentinel_not_flagged_when_column_has_other_negatives():
    # rest of the column (excluding the -999 occurrences) is not
    # non-negative, so -999 does not count as a sentinel here.
    series = pd.Series([-999, -5, 3, 7], name="col")
    result = detect_sentinels(series)
    assert result.has_sentinels is False
    assert result.sentinel_values_found == []


def test_numeric_sentinel_9999_flagged():
    series = pd.Series([1, 2, 3, 9999], name="col")
    result = detect_sentinels(series)
    assert "9999" in result.sentinel_values_found


def test_string_sentinels_detected_exactly():
    series = pd.Series(["x", "N/A", "", "valid", "NA"], name="col")
    result = detect_sentinels(series)
    assert result.has_sentinels is True
    assert set(result.sentinel_values_found) == {"N/A", "NA", '""'}
    assert result.affected_row_count == 3


def test_string_sentinels_are_case_sensitive_exact_matches():
    # "na" (lowercase) is not one of the literal spec tokens "NA"/"N/A"/""
    series = pd.Series(["na", "valid1", "valid2"], name="col")
    result = detect_sentinels(series)
    assert result.has_sentinels is False


def test_sentinel_detection_ignores_nulls():
    series = pd.Series([1, 2, None, 4], name="col")
    result = detect_sentinels(series)
    assert result.has_sentinels is False


def test_sentinel_detection_all_null_column():
    series = pd.Series([None, None], name="col")
    result = detect_sentinels(series)
    assert result.has_sentinels is False
    assert result.affected_row_count == 0


def test_sentinel_detection_uses_explicit_column_name_override():
    series = pd.Series([1, -999], name="original_name")
    result = detect_sentinels(series, column_name="override_name")
    assert result.column == "override_name"


# ---------------------------------------------------------------------------
# subgroup_reversal
# ---------------------------------------------------------------------------


def test_classic_simpsons_paradox_is_flagged():
    # Two subgroups, each with a perfect negative x-y slope, offset so the
    # combined (overall) relationship is positive.
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [10, 9, 8, 7, 14, 13, 12, 11]
    groups = ["A", "A", "A", "A", "B", "B", "B", "B"]

    result = detect_subgroup_reversal(x, y, groups)

    assert result.overall_sign == 1
    assert result.subgroup_signs["A"] == -1
    assert result.subgroup_signs["B"] == -1
    assert set(result.reversed_subgroups) == {"A", "B"}
    assert result.is_simpsons_paradox_risk is True


def test_agreeing_subgroups_are_not_flagged():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [2, 4, 5, 7, 9, 10, 13, 14]
    groups = ["A", "A", "A", "A", "B", "B", "B", "B"]

    result = detect_subgroup_reversal(x, y, groups)

    assert result.overall_sign == 1
    assert result.subgroup_signs["A"] == 1
    assert result.subgroup_signs["B"] == 1
    assert result.reversed_subgroups == []
    assert result.is_simpsons_paradox_risk is False


def test_subgroup_reversal_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        detect_subgroup_reversal([1, 2, 3], [1, 2], ["A", "B", "A"])


def test_subgroup_reversal_skips_undersized_groups():
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    groups = ["A", "A", "A", "A", "B"]  # "B" has only one observation

    result = detect_subgroup_reversal(x, y, groups)

    assert "B" not in result.subgroup_signs
    assert "A" in result.subgroup_signs
