"""
Bayesian Predictive Model for the Reversal Curse.

This module implements a computational model that:
1. Takes pre-reversal and post-reversal information structures as input
2. Predicts the magnitude of the reversal curse
3. Models the cognitive load during knowledge restructuring
4. Provides theoretical grounding for debiasing interventions

The model is based on Bayesian principles of belief updating,
where the "curse" emerges from asymmetric updating costs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.special import softmax
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeStructure:
    """
    Representation of a knowledge state before or after reversal.

    The knowledge structure captures:
    - Belief strengths for different propositions
    - Uncertainty/confidence levels
    - Interconnections between concepts
    - Temporal ordering of information acquisition
    """

    propositions: Dict[str, float]  # proposition -> belief strength [0, 1]
    uncertainties: Dict[str, float]  # proposition -> uncertainty level
    connections: Dict[Tuple[str, str], float]  # (prop_i, prop_j) -> connection strength
    acquisition_order: List[str]  # Order in which propositions were learned
    exposure_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def n_propositions(self) -> int:
        return len(self.propositions)

    @property
    def mean_belief_strength(self) -> float:
        if not self.propositions:
            return 0.0
        return np.mean(list(self.propositions.values()))

    @property
    def mean_uncertainty(self) -> float:
        if not self.uncertainties:
            return 1.0
        return np.mean(list(self.uncertainties.values()))

    @property
    def structural_coherence(self) -> float:
        """
        Measure how interconnected the knowledge structure is.
        Higher coherence = more integrated knowledge.
        """
        if not self.connections:
            return 0.0
        return np.mean(list(self.connections.values()))

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for model input."""
        features = [
            self.mean_belief_strength,
            self.mean_uncertainty,
            self.structural_coherence,
            self.n_propositions,
            len(self.connections),
            np.std(list(self.propositions.values())) if self.propositions else 0,
        ]
        return np.array(features)


@dataclass
class ReversalEvent:
    """
    Representation of a knowledge reversal event.

    Captures the nature and magnitude of the conceptual shift.
    """

    pre_state: KnowledgeStructure
    post_state: KnowledgeStructure
    reversal_type: str  # "contradiction", "refinement", "paradigm_shift"
    affected_propositions: List[str]
    reversal_strength: float  # 0 = minor update, 1 = complete reversal

    @property
    def structural_disruption(self) -> float:
        """
        Measure how much the reversal disrupts the knowledge structure.
        """
        # Compute change in belief strengths
        pre_beliefs = self.pre_state.propositions
        post_beliefs = self.post_state.propositions

        shared_props = set(pre_beliefs.keys()) & set(post_beliefs.keys())
        if not shared_props:
            return 1.0

        changes = [
            abs(pre_beliefs[p] - post_beliefs[p])
            for p in shared_props
        ]
        return np.mean(changes)

    @property
    def connection_disruption(self) -> float:
        """
        Measure how much inter-concept connections are affected.
        """
        pre_connections = set(self.pre_state.connections.keys())
        post_connections = set(self.post_state.connections.keys())

        # Jaccard distance
        union = pre_connections | post_connections
        if not union:
            return 0.0

        intersection = pre_connections & post_connections
        return 1.0 - len(intersection) / len(union)


class BayesianCurseModel:
    """
    Bayesian model for predicting reversal curse magnitude.

    The model assumes that the "curse" arises from:
    1. Asymmetric belief updating (easier to strengthen than weaken beliefs)
    2. Cognitive load during restructuring
    3. Theory of mind degradation under uncertainty
    4. Temporal encoding effects (what was learned first is harder to revise)

    The predicted curse magnitude can be validated against human performance.
    """

    def __init__(
        self,
        updating_asymmetry: float = 0.3,
        load_sensitivity: float = 0.5,
        tom_degradation_rate: float = 0.4,
        temporal_weight: float = 0.2,
        noise_level: float = 0.1
    ):
        """
        Initialize the model with parameters.

        Parameters
        ----------
        updating_asymmetry : float
            How much harder it is to weaken vs. strengthen beliefs (0-1)
        load_sensitivity : float
            How much cognitive load affects performance (0-1)
        tom_degradation_rate : float
            How much theory of mind degrades under uncertainty (0-1)
        temporal_weight : float
            Weight given to order of acquisition effects (0-1)
        noise_level : float
            Intrinsic noise in the system (0-1)
        """
        self.updating_asymmetry = updating_asymmetry
        self.load_sensitivity = load_sensitivity
        self.tom_degradation_rate = tom_degradation_rate
        self.temporal_weight = temporal_weight
        self.noise_level = noise_level

        # Fitted parameters (initialized to priors)
        self.fitted = False
        self.params_ = {
            "updating_asymmetry": updating_asymmetry,
            "load_sensitivity": load_sensitivity,
            "tom_degradation_rate": tom_degradation_rate,
            "temporal_weight": temporal_weight,
        }

    def compute_cognitive_load(self, reversal: ReversalEvent) -> float:
        """
        Compute the cognitive load induced by a reversal.

        Higher structural disruption and more affected propositions
        lead to higher cognitive load.
        """
        base_load = reversal.structural_disruption

        # Scale by number of affected propositions
        prop_factor = len(reversal.affected_propositions) / max(
            reversal.pre_state.n_propositions, 1
        )

        # Add connection disruption
        connection_factor = reversal.connection_disruption

        cognitive_load = (
            0.4 * base_load +
            0.3 * prop_factor +
            0.3 * connection_factor
        )

        return np.clip(cognitive_load, 0, 1)

    def compute_updating_cost(self, reversal: ReversalEvent) -> float:
        """
        Compute the asymmetric cost of belief updating.

        Weakening strong beliefs costs more than strengthening weak ones.
        """
        pre_beliefs = reversal.pre_state.propositions
        post_beliefs = reversal.post_state.propositions

        total_cost = 0
        n_updates = 0

        for prop in reversal.affected_propositions:
            if prop in pre_beliefs and prop in post_beliefs:
                delta = post_beliefs[prop] - pre_beliefs[prop]

                if delta < 0:  # Weakening belief
                    # Cost is higher for weakening, scaled by asymmetry parameter
                    cost = abs(delta) * (1 + self.params_["updating_asymmetry"])
                else:  # Strengthening belief
                    cost = abs(delta)

                total_cost += cost
                n_updates += 1

        if n_updates == 0:
            return 0.0

        return total_cost / n_updates

    def compute_temporal_penalty(self, reversal: ReversalEvent) -> float:
        """
        Compute penalty based on temporal order of acquisition.

        Propositions learned earlier are harder to revise.
        """
        acquisition_order = reversal.pre_state.acquisition_order
        affected = set(reversal.affected_propositions)

        if not acquisition_order or not affected:
            return 0.0

        penalties = []
        n_total = len(acquisition_order)

        for i, prop in enumerate(acquisition_order):
            if prop in affected:
                # Earlier positions (lower i) get higher penalty
                position_penalty = 1.0 - (i / n_total)
                penalties.append(position_penalty)

        if not penalties:
            return 0.0

        return np.mean(penalties)

    def compute_tom_degradation(
        self,
        reversal: ReversalEvent,
        cognitive_load: float
    ) -> float:
        """
        Compute theory of mind degradation.

        Under high cognitive load (post-reversal), the ability to
        model what a novice knows/doesn't know is impaired.
        """
        # Base degradation from uncertainty
        uncertainty_factor = reversal.post_state.mean_uncertainty

        # Degradation increases with cognitive load
        load_factor = cognitive_load * self.params_["tom_degradation_rate"]

        # Combine factors
        degradation = 0.5 * uncertainty_factor + 0.5 * load_factor

        return np.clip(degradation, 0, 1)

    def predict_curse_magnitude(
        self,
        reversal: ReversalEvent,
        return_components: bool = False
    ) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Predict the magnitude of the reversal curse.

        Parameters
        ----------
        reversal : ReversalEvent
            The reversal event to analyze
        return_components : bool
            Whether to return individual component scores

        Returns
        -------
        Union[float, Tuple[float, Dict]]
            Predicted curse magnitude (0-1) and optionally component scores
        """
        # Compute component factors
        cognitive_load = self.compute_cognitive_load(reversal)
        updating_cost = self.compute_updating_cost(reversal)
        temporal_penalty = self.compute_temporal_penalty(reversal)
        tom_degradation = self.compute_tom_degradation(reversal, cognitive_load)

        # Combine into overall curse magnitude
        curse_magnitude = (
            self.params_["load_sensitivity"] * cognitive_load +
            0.3 * updating_cost +
            self.params_["temporal_weight"] * temporal_penalty +
            0.3 * tom_degradation
        )

        # Add noise
        if self.noise_level > 0:
            curse_magnitude += np.random.normal(0, self.noise_level)

        curse_magnitude = np.clip(curse_magnitude, 0, 1)

        if return_components:
            components = {
                "cognitive_load": cognitive_load,
                "updating_cost": updating_cost,
                "temporal_penalty": temporal_penalty,
                "tom_degradation": tom_degradation,
            }
            return curse_magnitude, components

        return curse_magnitude

    def fit(
        self,
        reversals: List[ReversalEvent],
        observed_magnitudes: np.ndarray,
        method: str = "L-BFGS-B"
    ) -> Dict[str, Any]:
        """
        Fit model parameters to observed data.

        Parameters
        ----------
        reversals : List[ReversalEvent]
            List of reversal events
        observed_magnitudes : np.ndarray
            Observed curse magnitudes for each event
        method : str
            Optimization method

        Returns
        -------
        Dict[str, Any]
            Fitting results including optimized parameters
        """
        def loss_function(params):
            self.params_["updating_asymmetry"] = params[0]
            self.params_["load_sensitivity"] = params[1]
            self.params_["tom_degradation_rate"] = params[2]
            self.params_["temporal_weight"] = params[3]

            predictions = [
                self.predict_curse_magnitude(r)
                for r in reversals
            ]

            mse = np.mean((np.array(predictions) - observed_magnitudes) ** 2)
            return mse

        # Initial parameter values
        x0 = [
            self.updating_asymmetry,
            self.load_sensitivity,
            self.tom_degradation_rate,
            self.temporal_weight,
        ]

        # Parameter bounds
        bounds = [(0, 1)] * 4

        # Optimize
        result = minimize(
            loss_function,
            x0,
            method=method,
            bounds=bounds
        )

        # Store fitted parameters
        self.params_["updating_asymmetry"] = result.x[0]
        self.params_["load_sensitivity"] = result.x[1]
        self.params_["tom_degradation_rate"] = result.x[2]
        self.params_["temporal_weight"] = result.x[3]
        self.fitted = True

        # Compute final predictions
        final_predictions = [
            self.predict_curse_magnitude(r)
            for r in reversals
        ]

        # Compute fit statistics
        correlation = np.corrcoef(final_predictions, observed_magnitudes)[0, 1]
        r_squared = correlation ** 2
        rmse = np.sqrt(result.fun)

        return {
            "success": result.success,
            "parameters": dict(self.params_),
            "r_squared": r_squared,
            "rmse": rmse,
            "correlation": correlation,
            "n_iterations": result.nit,
            "predictions": final_predictions,
        }

    def predict_intervention_effect(
        self,
        reversal: ReversalEvent,
        intervention_strength: float
    ) -> float:
        """
        Predict how much an intervention would reduce the curse.

        The intervention is modeled as reducing cognitive load
        and improving theory of mind maintenance.

        Parameters
        ----------
        reversal : ReversalEvent
            The reversal event
        intervention_strength : float
            Strength of the intervention (0-1)

        Returns
        -------
        float
            Predicted reduction in curse magnitude
        """
        # Baseline curse
        baseline, components = self.predict_curse_magnitude(
            reversal, return_components=True
        )

        # Intervention reduces cognitive load and ToM degradation
        reduced_load = components["cognitive_load"] * (1 - 0.6 * intervention_strength)
        reduced_tom = components["tom_degradation"] * (1 - 0.5 * intervention_strength)

        # Recalculate curse with intervention
        intervened_curse = (
            self.params_["load_sensitivity"] * reduced_load +
            0.3 * components["updating_cost"] +
            self.params_["temporal_weight"] * components["temporal_penalty"] +
            0.3 * reduced_tom
        )

        reduction = baseline - intervened_curse
        return max(0, reduction)


def predict_curse_magnitude(
    pre_beliefs: Dict[str, float],
    post_beliefs: Dict[str, float],
    affected_propositions: List[str],
    reversal_strength: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """
    Convenience function to predict curse magnitude from belief structures.

    Parameters
    ----------
    pre_beliefs : Dict[str, float]
        Beliefs before reversal
    post_beliefs : Dict[str, float]
        Beliefs after reversal
    affected_propositions : List[str]
        Which propositions were affected
    reversal_strength : float
        Overall strength of the reversal

    Returns
    -------
    Tuple[float, Dict[str, float]]
        Predicted curse magnitude and component scores
    """
    # Construct knowledge structures
    pre_state = KnowledgeStructure(
        propositions=pre_beliefs,
        uncertainties={p: 0.2 for p in pre_beliefs},
        connections={},
        acquisition_order=list(pre_beliefs.keys()),
    )

    post_state = KnowledgeStructure(
        propositions=post_beliefs,
        uncertainties={p: 0.5 for p in post_beliefs},  # Higher uncertainty post-reversal
        connections={},
        acquisition_order=list(post_beliefs.keys()),
    )

    reversal = ReversalEvent(
        pre_state=pre_state,
        post_state=post_state,
        reversal_type="contradiction",
        affected_propositions=affected_propositions,
        reversal_strength=reversal_strength,
    )

    model = BayesianCurseModel()
    return model.predict_curse_magnitude(reversal, return_components=True)


def create_simulated_reversal(
    n_propositions: int = 10,
    reversal_fraction: float = 0.5,
    seed: Optional[int] = None
) -> ReversalEvent:
    """
    Create a simulated reversal event for testing.

    Parameters
    ----------
    n_propositions : int
        Number of propositions in the knowledge structure
    reversal_fraction : float
        Fraction of propositions affected by reversal
    seed : Optional[int]
        Random seed

    Returns
    -------
    ReversalEvent
        Simulated reversal event
    """
    rng = np.random.default_rng(seed)

    # Create proposition names
    props = [f"P{i}" for i in range(n_propositions)]

    # Pre-reversal beliefs (high confidence)
    pre_beliefs = {p: rng.uniform(0.7, 0.95) for p in props}
    pre_uncertainties = {p: rng.uniform(0.1, 0.3) for p in props}

    # Select affected propositions
    n_affected = max(1, int(n_propositions * reversal_fraction))
    affected = list(rng.choice(props, size=n_affected, replace=False))

    # Post-reversal beliefs (reversed for affected, same for others)
    post_beliefs = {}
    for p in props:
        if p in affected:
            # Reverse the belief
            post_beliefs[p] = 1.0 - pre_beliefs[p] + rng.uniform(-0.1, 0.1)
            post_beliefs[p] = np.clip(post_beliefs[p], 0, 1)
        else:
            post_beliefs[p] = pre_beliefs[p]

    post_uncertainties = {
        p: 0.6 if p in affected else pre_uncertainties[p]
        for p in props
    }

    # Create connections (random subset)
    connections = {}
    for i, p1 in enumerate(props):
        for p2 in props[i+1:]:
            if rng.random() < 0.3:
                connections[(p1, p2)] = rng.uniform(0.3, 0.8)

    pre_state = KnowledgeStructure(
        propositions=pre_beliefs,
        uncertainties=pre_uncertainties,
        connections=connections,
        acquisition_order=props,
    )

    # Post-reversal connections may be disrupted
    post_connections = {
        k: v * (0.5 if k[0] in affected or k[1] in affected else 1.0)
        for k, v in connections.items()
    }

    post_state = KnowledgeStructure(
        propositions=post_beliefs,
        uncertainties=post_uncertainties,
        connections=post_connections,
        acquisition_order=props,
    )

    return ReversalEvent(
        pre_state=pre_state,
        post_state=post_state,
        reversal_type="contradiction",
        affected_propositions=affected,
        reversal_strength=reversal_fraction,
    )
