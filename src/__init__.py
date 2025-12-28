"""
Reversal Curse Research Project

A comprehensive framework for investigating the reversal curse in human memory,
providing evidence that memory encodes directional predictions rather than
bidirectional associations.

Studies:
    - Study 1: Large-scale Duolingo observational analysis
    - Study 2: Wikipedia/Wikidata factual knowledge analysis
    - Study 3: Controlled experimental manipulation of presentation order

Modules:
    - analysis: Statistical analysis pipelines
    - data: Data loading and preprocessing
    - experiment: Experimental platform components
    - visualization: Figure generation for publication
    - utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from config.settings import get_config, config

__all__ = [
    "__version__",
    "get_config",
    "config",
]
