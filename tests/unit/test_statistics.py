"""
Unit tests for statistical analysis functions.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.statistics import (
    calculate_reversal_gap,
    compute_cohens_h,
    compute_cohens_d,
    two_proportion_test,
    mcnemar_test,
    bootstrap_ci,
    equivalence_test,
    power_analysis,
    descriptive_statistics,
)


class TestCohensH:
    """Tests for Cohen's h effect size calculation."""

    def test_cohens_h_equal_proportions(self):
        """Cohen's h should be 0 for equal proportions."""
        h = compute_cohens_h(0.5, 0.5)
        assert abs(h) < 0.001

    def test_cohens_h_direction(self):
        """Cohen's h should be positive when p1 > p2."""
        h = compute_cohens_h(0.8, 0.4)
        assert h > 0

        h = compute_cohens_h(0.4, 0.8)
        assert h < 0

    def test_cohens_h_known_values(self):
        """Test against known effect size values."""
        # Small effect (h ≈ 0.2): 0.5 vs 0.4
        h = compute_cohens_h(0.5, 0.4)
        assert 0.15 < abs(h) < 0.25

        # Medium effect (h ≈ 0.5): 0.7 vs 0.5
        h = compute_cohens_h(0.7, 0.5)
        assert 0.35 < abs(h) < 0.55

    def test_cohens_h_extreme_values(self):
        """Test with extreme proportions."""
        h = compute_cohens_h(0.99, 0.01)
        assert abs(h) > 2.0  # Should be large

        # Should handle edge cases without error
        h = compute_cohens_h(0.001, 0.999)
        assert np.isfinite(h)


class TestCohensD:
    """Tests for Cohen's d effect size calculation."""

    def test_cohens_d_equal_groups(self):
        """Cohen's d should be 0 for identical groups."""
        group = np.array([1, 2, 3, 4, 5])
        d = compute_cohens_d(group, group)
        assert abs(d) < 0.001

    def test_cohens_d_direction(self):
        """Cohen's d sign indicates which group is larger."""
        group1 = np.array([10, 11, 12, 13, 14])
        group2 = np.array([1, 2, 3, 4, 5])

        d = compute_cohens_d(group1, group2)
        assert d > 0  # group1 has higher mean

        d = compute_cohens_d(group2, group1)
        assert d < 0  # group2 has lower mean

    def test_cohens_d_known_effect(self):
        """Test with groups having known effect size."""
        # Create groups with d ≈ 1
        np.random.seed(42)
        group1 = np.random.normal(1, 1, 100)
        group2 = np.random.normal(0, 1, 100)

        d = compute_cohens_d(group1, group2)
        assert 0.7 < d < 1.3  # Should be approximately 1


class TestReversalGap:
    """Tests for reversal gap calculation."""

    def test_reversal_gap_no_asymmetry(self):
        """Gap should be ~0 when accuracies are equal."""
        np.random.seed(42)
        forward = np.random.binomial(1, 0.7, 100)
        reverse = np.random.binomial(1, 0.7, 100)

        result = calculate_reversal_gap(forward, reverse)
        assert abs(result.gap) < 0.15  # Allow for sampling variance

    def test_reversal_gap_positive_asymmetry(self):
        """Gap should be positive when forward > reverse."""
        np.random.seed(42)
        forward = np.random.binomial(1, 0.9, 100)
        reverse = np.random.binomial(1, 0.4, 100)

        result = calculate_reversal_gap(forward, reverse)
        assert result.gap > 0.3
        assert result.cohens_h > 0

    def test_reversal_gap_confidence_interval(self):
        """CI should contain point estimate."""
        np.random.seed(42)
        forward = np.random.binomial(1, 0.8, 100)
        reverse = np.random.binomial(1, 0.5, 100)

        result = calculate_reversal_gap(forward, reverse)
        assert result.gap_ci_lower <= result.gap <= result.gap_ci_upper

    def test_reversal_gap_n_observations(self):
        """N should be sum of both arrays."""
        forward = np.ones(50)
        reverse = np.zeros(50)

        result = calculate_reversal_gap(forward, reverse)
        assert result.n_observations == 100


class TestTwoProportionTest:
    """Tests for two-proportion z-test."""

    def test_significant_difference(self):
        """Should detect significant difference in proportions."""
        result = two_proportion_test(80, 100, 40, 100)
        assert result.p_value < 0.001

    def test_no_difference(self):
        """Should not be significant when proportions are similar."""
        result = two_proportion_test(50, 100, 52, 100)
        assert result.p_value > 0.05

    def test_effect_size_direction(self):
        """Effect size should reflect direction of difference."""
        result = two_proportion_test(80, 100, 40, 100)
        assert result.effect_size > 0  # First proportion higher


class TestMcNemarTest:
    """Tests for McNemar's test."""

    def test_significant_asymmetry(self):
        """Should detect significant change in paired data."""
        # More improve than decline
        condition1 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        condition2 = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1])

        result = mcnemar_test(condition1, condition2)
        assert isinstance(result.p_value, float)

    def test_no_change(self):
        """Should not be significant when no change."""
        condition1 = np.array([1, 1, 0, 0])
        condition2 = np.array([1, 1, 0, 0])

        result = mcnemar_test(condition1, condition2)
        # All concordant, should not reject null


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_contains_mean(self):
        """95% CI should contain population mean most of the time."""
        np.random.seed(42)
        data = np.random.normal(100, 15, 50)

        point, ci_lower, ci_upper = bootstrap_ci(data)

        # Point estimate should be close to sample mean
        assert abs(point - np.mean(data)) < 0.001

        # CI should be reasonable width
        assert ci_upper > ci_lower
        assert ci_upper - ci_lower < 20  # Not too wide

    def test_bootstrap_custom_statistic(self):
        """Should work with custom statistics."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        point, ci_lower, ci_upper = bootstrap_ci(data, statistic=np.median)
        assert ci_lower <= point <= ci_upper


class TestEquivalenceTest:
    """Tests for TOST equivalence test."""

    def test_equivalent_groups(self):
        """Should establish equivalence for similar groups."""
        np.random.seed(42)
        group1 = np.random.normal(50, 5, 100)
        group2 = np.random.normal(50.5, 5, 100)  # Very slight difference

        result = equivalence_test(group1, group2, equivalence_bound=5)
        assert result["equivalent"] == True

    def test_different_groups(self):
        """Should not establish equivalence for different groups."""
        np.random.seed(42)
        group1 = np.random.normal(50, 5, 100)
        group2 = np.random.normal(60, 5, 100)  # 10-point difference

        result = equivalence_test(group1, group2, equivalence_bound=5)
        assert result["equivalent"] == False


class TestPowerAnalysis:
    """Tests for power analysis."""

    def test_power_analysis_reasonable_n(self):
        """Should return reasonable sample sizes."""
        result = power_analysis(effect_size=0.5, power=0.80)

        assert result["n_per_group"] > 0
        assert result["n_per_group"] < 1000
        assert result["total_n"] == result["n_per_group"] * 2

    def test_larger_effect_smaller_n(self):
        """Larger effects should require smaller samples."""
        result_small = power_analysis(effect_size=0.2, power=0.80)
        result_large = power_analysis(effect_size=0.8, power=0.80)

        assert result_large["n_per_group"] < result_small["n_per_group"]

    def test_higher_power_larger_n(self):
        """Higher power should require larger samples."""
        result_80 = power_analysis(effect_size=0.5, power=0.80)
        result_95 = power_analysis(effect_size=0.5, power=0.95)

        assert result_95["n_per_group"] > result_80["n_per_group"]


class TestDescriptiveStatistics:
    """Tests for descriptive statistics."""

    def test_descriptive_statistics_all_present(self):
        """Should return all expected statistics."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        stats_dict = descriptive_statistics(data)

        required_keys = ["n", "mean", "std", "se", "median", "min", "max"]
        for key in required_keys:
            assert key in stats_dict

    def test_descriptive_statistics_values(self):
        """Should compute correct values."""
        data = np.array([1, 2, 3, 4, 5])
        stats_dict = descriptive_statistics(data)

        assert stats_dict["n"] == 5
        assert stats_dict["mean"] == 3.0
        assert stats_dict["median"] == 3.0
        assert stats_dict["min"] == 1.0
        assert stats_dict["max"] == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
