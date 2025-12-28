"""
Visualization module for the Reversal Curse research project.

This module provides publication-ready figure generation for:
- Nature / Nature Human Behaviour formatting
- Main result figures
- Supplementary figures
- Statistical result visualizations
"""

from .figures import (
    FigureGenerator,
    create_empty_triangle_plot,
    create_domain_comparison_plot,
    create_flip_plot,
    create_asymmetry_comparison_plot,
    create_effect_size_comparison,
)

__all__ = [
    "FigureGenerator",
    "create_empty_triangle_plot",
    "create_domain_comparison_plot",
    "create_flip_plot",
    "create_asymmetry_comparison_plot",
    "create_effect_size_comparison",
]
