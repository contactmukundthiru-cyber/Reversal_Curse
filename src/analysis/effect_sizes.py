"""
Effect size calculations and comparisons for the Reversal Curse research.

This module provides:
- Effect size calculations for various metrics
- Comparison to benchmark effects in memory literature
- Publication-ready effect size tables
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .statistics import compute_cohens_h, compute_cohens_d


@dataclass
class BenchmarkEffect:
    """A benchmark effect from the literature for comparison."""

    name: str
    cohens_h: float
    cohens_d: Optional[float]
    source: str
    year: int
    domain: str


# Benchmark effects from memory research literature
MEMORY_BENCHMARKS: List[BenchmarkEffect] = [
    BenchmarkEffect(
        name="Stroop effect",
        cohens_h=0.80,
        cohens_d=1.00,
        source="MacLeod (1991)",
        year=1991,
        domain="attention",
    ),
    BenchmarkEffect(
        name="Serial position (primacy)",
        cohens_h=0.73,
        cohens_d=0.85,
        source="Murdock (1962)",
        year=1962,
        domain="memory",
    ),
    BenchmarkEffect(
        name="Testing effect",
        cohens_h=0.65,
        cohens_d=0.70,
        source="Roediger & Karpicke (2006)",
        year=2006,
        domain="learning",
    ),
    BenchmarkEffect(
        name="Generation effect",
        cohens_h=0.61,
        cohens_d=0.65,
        source="Slamecka & Graf (1978)",
        year=1978,
        domain="memory",
    ),
    BenchmarkEffect(
        name="Spacing effect",
        cohens_h=0.58,
        cohens_d=0.60,
        source="Cepeda et al. (2006)",
        year=2006,
        domain="learning",
    ),
    BenchmarkEffect(
        name="Picture superiority",
        cohens_h=0.55,
        cohens_d=0.58,
        source="Paivio (1971)",
        year=1971,
        domain="memory",
    ),
    BenchmarkEffect(
        name="Levels of processing",
        cohens_h=0.50,
        cohens_d=0.55,
        source="Craik & Tulving (1975)",
        year=1975,
        domain="memory",
    ),
    BenchmarkEffect(
        name="Face-name asymmetry",
        cohens_h=0.45,
        cohens_d=0.48,
        source="McWeeny et al. (1987)",
        year=1987,
        domain="memory",
    ),
    BenchmarkEffect(
        name="Retrieval-induced forgetting",
        cohens_h=0.40,
        cohens_d=0.42,
        source="Anderson et al. (1994)",
        year=1994,
        domain="memory",
    ),
]


class EffectSizeCalculator:
    """
    Calculator for effect sizes with comparison to benchmarks.

    This class provides methods for:
    - Computing effect sizes from raw data
    - Comparing to literature benchmarks
    - Generating publication-ready tables
    """

    def __init__(self, benchmarks: Optional[List[BenchmarkEffect]] = None):
        """
        Initialize the calculator.

        Parameters
        ----------
        benchmarks : Optional[List[BenchmarkEffect]]
            Custom benchmark effects. Uses MEMORY_BENCHMARKS if None.
        """
        self.benchmarks = benchmarks or MEMORY_BENCHMARKS

    def compute_all_effect_sizes(
        self,
        p1: float,
        p2: float,
        n1: int,
        n2: int
    ) -> Dict[str, float]:
        """
        Compute multiple effect size metrics.

        Parameters
        ----------
        p1 : float
            Proportion in group 1
        p2 : float
            Proportion in group 2
        n1 : int
            Sample size group 1
        n2 : int
            Sample size group 2

        Returns
        -------
        Dict[str, float]
            Dictionary of effect sizes
        """
        cohens_h = compute_cohens_h(p1, p2)

        # Odds ratio
        odds1 = p1 / (1 - p1) if p1 < 1 else float('inf')
        odds2 = p2 / (1 - p2) if p2 < 1 else float('inf')
        odds_ratio = odds1 / odds2 if odds2 > 0 else float('inf')

        # Log odds ratio
        log_odds_ratio = np.log(odds_ratio) if odds_ratio > 0 and odds_ratio < float('inf') else np.nan

        # Risk ratio (relative risk)
        risk_ratio = p1 / p2 if p2 > 0 else float('inf')

        # Risk difference (absolute risk reduction)
        risk_difference = p1 - p2

        # Number needed to treat (if applicable)
        nnt = 1 / abs(risk_difference) if risk_difference != 0 else float('inf')

        # Phi coefficient (for 2x2 contingency)
        # Using approximation from proportions
        phi = self._phi_from_proportions(p1, p2, n1, n2)

        return {
            "cohens_h": cohens_h,
            "odds_ratio": odds_ratio,
            "log_odds_ratio": log_odds_ratio,
            "risk_ratio": risk_ratio,
            "risk_difference": risk_difference,
            "nnt": nnt,
            "phi": phi,
        }

    def _phi_from_proportions(
        self,
        p1: float,
        p2: float,
        n1: int,
        n2: int
    ) -> float:
        """Approximate phi coefficient from proportions."""
        # Create 2x2 table
        a = int(p1 * n1)  # Group 1 success
        b = n1 - a        # Group 1 failure
        c = int(p2 * n2)  # Group 2 success
        d = n2 - c        # Group 2 failure

        n = a + b + c + d
        numerator = (a * d) - (b * c)
        denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def interpret_cohens_h(self, h: float) -> str:
        """
        Interpret Cohen's h using standard conventions.

        Parameters
        ----------
        h : float
            Cohen's h value

        Returns
        -------
        str
            Interpretation string
        """
        abs_h = abs(h)

        if abs_h < 0.2:
            return "negligible"
        elif abs_h < 0.5:
            return "small"
        elif abs_h < 0.8:
            return "medium"
        else:
            return "large"

    def rank_against_benchmarks(
        self,
        effect_size: float,
        metric: str = "cohens_h"
    ) -> Tuple[int, int, str]:
        """
        Rank an effect size against benchmark effects.

        Parameters
        ----------
        effect_size : float
            The effect size to rank
        metric : str, default="cohens_h"
            Which metric to compare

        Returns
        -------
        Tuple[int, int, str]
            (rank, total_benchmarks, narrative)
        """
        abs_effect = abs(effect_size)

        if metric == "cohens_h":
            benchmark_values = [b.cohens_h for b in self.benchmarks]
        else:
            benchmark_values = [
                b.cohens_d for b in self.benchmarks
                if b.cohens_d is not None
            ]

        # Sort benchmarks descending
        sorted_values = sorted(benchmark_values, reverse=True)

        # Find rank
        rank = 1
        for val in sorted_values:
            if abs_effect > val:
                break
            rank += 1

        total = len(sorted_values) + 1  # Including our effect

        if rank == 1:
            narrative = "larger than all benchmark effects"
        elif rank <= 3:
            narrative = "among the largest effects in memory research"
        elif rank <= len(sorted_values) // 2:
            narrative = "above median for memory effects"
        else:
            narrative = "comparable to typical memory effects"

        return rank, total, narrative

    def generate_comparison_table(
        self,
        our_effect: float,
        our_effect_name: str = "Reversal Curse"
    ) -> List[Dict[str, any]]:
        """
        Generate a comparison table for publication.

        Parameters
        ----------
        our_effect : float
            Our effect size (Cohen's h)
        our_effect_name : str
            Name for our effect

        Returns
        -------
        List[Dict]
            Table data for visualization
        """
        table = []

        # Add benchmarks
        for benchmark in self.benchmarks:
            table.append({
                "name": benchmark.name,
                "cohens_h": benchmark.cohens_h,
                "source": benchmark.source,
                "year": benchmark.year,
                "domain": benchmark.domain,
                "is_our_effect": False,
            })

        # Add our effect
        table.append({
            "name": our_effect_name,
            "cohens_h": abs(our_effect),
            "source": "This paper",
            "year": 2024,
            "domain": "memory",
            "is_our_effect": True,
        })

        # Sort by effect size
        table.sort(key=lambda x: x["cohens_h"], reverse=True)

        return table


def compare_to_benchmarks(
    effect_size: float,
    effect_name: str = "Reversal Curse"
) -> Dict[str, any]:
    """
    Convenience function to compare an effect to benchmarks.

    Parameters
    ----------
    effect_size : float
        The effect size to compare
    effect_name : str
        Name of the effect

    Returns
    -------
    Dict
        Comparison results
    """
    calculator = EffectSizeCalculator()

    rank, total, narrative = calculator.rank_against_benchmarks(effect_size)
    interpretation = calculator.interpret_cohens_h(effect_size)
    table = calculator.generate_comparison_table(effect_size, effect_name)

    return {
        "effect_size": effect_size,
        "effect_name": effect_name,
        "interpretation": interpretation,
        "rank": rank,
        "total_compared": total,
        "narrative": narrative,
        "comparison_table": table,
    }


def cohens_h_ci(
    p1: float,
    p2: float,
    n1: int,
    n2: int,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Compute Cohen's h with confidence interval.

    Parameters
    ----------
    p1 : float
        Proportion in group 1
    p2 : float
        Proportion in group 2
    n1 : int
        Sample size group 1
    n2 : int
        Sample size group 2
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)
    """
    h = compute_cohens_h(p1, p2)

    # Standard error of h (using delta method approximation)
    # SE(h) ≈ sqrt(1/n1 + 1/n2)
    se_h = np.sqrt(1 / n1 + 1 / n2)

    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = h - z * se_h
    ci_upper = h + z * se_h

    return h, ci_lower, ci_upper
