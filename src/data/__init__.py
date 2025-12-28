"""
Data module for the Reversal Curse research project.

This module provides:
- Database models for experimental data
- Data loading and preprocessing utilities
- Export and import functionality
"""

from .models import (
    db,
    Participant,
    Trial,
    ExperimentSession,
    Stimulus,
    StudyConfiguration,
)

from .preprocessing import (
    preprocess_duolingo_data,
    preprocess_wikipedia_data,
    validate_experimental_data,
)

__all__ = [
    "db",
    "Participant",
    "Trial",
    "ExperimentSession",
    "Stimulus",
    "StudyConfiguration",
    "preprocess_duolingo_data",
    "preprocess_wikipedia_data",
    "validate_experimental_data",
]
