"""
Computational models for the Reversal Curse phenomenon.

This module provides:
- Bayesian predictive models of curse magnitude
- Neural network models for knowledge restructuring
- Intervention efficacy models
- Theory of mind computational frameworks
"""

from .bayesian import (
    BayesianCurseModel,
    KnowledgeStructure,
    predict_curse_magnitude,
)

from .neural_network import (
    ReversalCurseNet,
    train_predictor,
    evaluate_model,
)

from .intervention import (
    CognitiveScaffold,
    InterventionSimulator,
    compute_debiasing_effect,
)

__all__ = [
    "BayesianCurseModel",
    "KnowledgeStructure",
    "predict_curse_magnitude",
    "ReversalCurseNet",
    "train_predictor",
    "evaluate_model",
    "CognitiveScaffold",
    "InterventionSimulator",
    "compute_debiasing_effect",
]
