"""
Analysis module for the Reversal Curse research project.

This module provides statistical analysis tools for all three studies:
- Duolingo large-scale observational analysis
- Wikipedia/Wikidata factual knowledge analysis
- Controlled experimental analysis

Key components:
    - statistics: Core statistical functions and tests
    - duolingo: Duolingo-specific analysis pipeline
    - wikipedia: Wikipedia/Wikidata analysis pipeline
    - experimental: Controlled experiment analysis
    - effect_sizes: Effect size calculations and comparisons
"""

from .statistics import (
    calculate_reversal_gap,
    mixed_effects_model,
    compute_cohens_h,
    bootstrap_ci,
    equivalence_test,
    power_analysis,
)

from .effect_sizes import (
    EffectSizeCalculator,
    compare_to_benchmarks,
)

__all__ = [
    "calculate_reversal_gap",
    "mixed_effects_model",
    "compute_cohens_h",
    "bootstrap_ci",
    "equivalence_test",
    "power_analysis",
    "EffectSizeCalculator",
    "compare_to_benchmarks",
]
