"""
Experiment module for the Reversal Curse research project.

This module provides:
- Stimulus generation
- Experiment protocol management
- Prolific integration
- Data validation
"""

from .stimuli import StimulusGenerator, Symbol, Label

__all__ = [
    "StimulusGenerator",
    "Symbol",
    "Label",
]
