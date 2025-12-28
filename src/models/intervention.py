"""
Debiasing Intervention Module for the Reversal Curse.

This module provides computational models for interventions that can
reduce the reversal curse effect. The key insight is that forcing
explicit knowledge restructuring can mitigate the curse.

Key components:
1. CognitiveScaffold: Interactive framework for knowledge mapping
2. InterventionSimulator: Predicts intervention efficacy
3. Debiasing strategies based on computational model principles
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta as beta_dist

from .bayesian import BayesianCurseModel, KnowledgeStructure, ReversalEvent

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of debiasing interventions."""

    EXPLICIT_MAPPING = "explicit_mapping"  # Force explicit old->new mapping
    PERSPECTIVE_TAKING = "perspective_taking"  # Simulate novice perspective
    UNCERTAINTY_HIGHLIGHTING = "uncertainty_highlighting"  # Emphasize post-reversal uncertainty
    TEMPORAL_REFRAMING = "temporal_reframing"  # Reframe acquisition order
    STRUCTURAL_VISUALIZATION = "structural_visualization"  # Visualize knowledge changes
    INCREMENTAL_BRIDGING = "incremental_bridging"  # Build bridge between old/new understanding


@dataclass
class InterventionConfig:
    """Configuration for intervention parameters."""

    intensity: float = 0.7  # 0-1, how intensive the intervention
    duration_minutes: int = 10  # Time allocated for intervention
    n_mapping_prompts: int = 5  # Number of explicit mapping prompts
    include_reflection: bool = True  # Include metacognitive reflection
    feedback_mode: str = "adaptive"  # "fixed", "adaptive", or "none"
    personalized: bool = True  # Adapt to individual characteristics


@dataclass
class MappingPrompt:
    """A prompt for explicit knowledge mapping."""

    prompt_text: str
    old_concept: str
    new_concept: str
    requires_explanation: bool = True
    difficulty_level: float = 0.5  # 0-1


@dataclass
class InterventionResult:
    """Result of applying an intervention."""

    baseline_curse: float
    post_intervention_curse: float
    reduction_magnitude: float
    reduction_percent: float
    confidence_interval: Tuple[float, float]
    component_effects: Dict[str, float]
    participant_engagement: float
    time_spent_seconds: int


class CognitiveScaffold:
    """
    Cognitive scaffold for explicit knowledge restructuring.

    This intervention forces experts to explicitly map their old
    understanding to their new understanding before communicating
    with novices. The scaffold:

    1. Identifies key propositions that changed
    2. Generates mapping prompts
    3. Tracks completion and quality of mappings
    4. Provides feedback on perspective-taking accuracy
    """

    def __init__(self, config: Optional[InterventionConfig] = None):
        """
        Initialize the cognitive scaffold.

        Parameters
        ----------
        config : InterventionConfig, optional
            Intervention configuration
        """
        self.config = config or InterventionConfig()
        self.active_session = False
        self.mappings: List[Dict[str, Any]] = []
        self.session_history: List[Dict[str, Any]] = []

    def start_session(
        self,
        reversal: ReversalEvent,
        participant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start a scaffolding session for a knowledge reversal.

        Parameters
        ----------
        reversal : ReversalEvent
            The reversal event to scaffold
        participant_id : str, optional
            Identifier for the participant

        Returns
        -------
        Dict[str, Any]
            Session information including initial prompts
        """
        self.active_session = True
        self.current_reversal = reversal
        self.mappings = []

        # Generate mapping prompts
        prompts = self._generate_mapping_prompts(reversal)

        # Initialize session
        session = {
            "participant_id": participant_id,
            "reversal_strength": reversal.reversal_strength,
            "n_affected_propositions": len(reversal.affected_propositions),
            "prompts": prompts,
            "current_prompt_idx": 0,
            "completed_mappings": 0,
            "start_time": None,  # To be set by caller
        }

        return session

    def _generate_mapping_prompts(
        self,
        reversal: ReversalEvent
    ) -> List[MappingPrompt]:
        """Generate prompts for explicit mapping."""
        prompts = []

        pre_beliefs = reversal.pre_state.propositions
        post_beliefs = reversal.post_state.propositions

        # Sort affected propositions by magnitude of change
        changes = []
        for prop in reversal.affected_propositions:
            if prop in pre_beliefs and prop in post_beliefs:
                change = abs(post_beliefs[prop] - pre_beliefs[prop])
                changes.append((prop, change))

        changes.sort(key=lambda x: x[1], reverse=True)

        # Generate prompts for top changes
        n_prompts = min(self.config.n_mapping_prompts, len(changes))

        prompt_templates = [
            "Explain how your understanding of {concept} changed from {old_state} to {new_state}.",
            "A newcomer to this field would expect {old_state} about {concept}. Explain why {new_state} is actually the case.",
            "Before the new information, experts believed {old_state}. Now we understand {new_state}. What led to this change?",
            "Compare and contrast the old view ({old_state}) with the new view ({new_state}) regarding {concept}.",
            "Why might someone still believe {old_state} about {concept}? How would you explain that {new_state} is correct?",
        ]

        for i, (prop, change) in enumerate(changes[:n_prompts]):
            old_state = self._belief_to_description(pre_beliefs[prop])
            new_state = self._belief_to_description(post_beliefs[prop])

            template = prompt_templates[i % len(prompt_templates)]
            prompt_text = template.format(
                concept=prop,
                old_state=old_state,
                new_state=new_state
            )

            prompts.append(MappingPrompt(
                prompt_text=prompt_text,
                old_concept=f"{prop}:{old_state}",
                new_concept=f"{prop}:{new_state}",
                requires_explanation=True,
                difficulty_level=change  # Higher change = more difficult
            ))

        return prompts

    def _belief_to_description(self, belief_strength: float) -> str:
        """Convert belief strength to natural language description."""
        if belief_strength > 0.8:
            return "strongly believed/confirmed"
        elif belief_strength > 0.6:
            return "moderately believed/likely"
        elif belief_strength > 0.4:
            return "uncertain/debated"
        elif belief_strength > 0.2:
            return "unlikely/doubted"
        else:
            return "strongly disbelieved/rejected"

    def record_mapping(
        self,
        prompt_idx: int,
        response: str,
        response_time_seconds: float,
        quality_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Record a participant's mapping response.

        Parameters
        ----------
        prompt_idx : int
            Index of the prompt being responded to
        response : str
            Participant's response
        response_time_seconds : float
            Time taken to respond
        quality_score : float, optional
            External quality assessment (0-1)

        Returns
        -------
        Dict[str, Any]
            Mapping record with feedback
        """
        if not self.active_session:
            raise RuntimeError("No active session")

        # Auto-assess quality if not provided
        if quality_score is None:
            quality_score = self._assess_response_quality(response)

        mapping = {
            "prompt_idx": prompt_idx,
            "response": response,
            "response_time": response_time_seconds,
            "quality_score": quality_score,
            "word_count": len(response.split()),
            "mentions_both_states": self._check_dual_mention(response, prompt_idx),
        }

        self.mappings.append(mapping)

        # Generate feedback
        feedback = self._generate_feedback(mapping)
        mapping["feedback"] = feedback

        return mapping

    def _assess_response_quality(self, response: str) -> float:
        """
        Assess response quality using heuristics.

        Real implementation would use NLP/LLM assessment.
        """
        score = 0.0

        # Length factor (longer = more thorough, up to a point)
        word_count = len(response.split())
        length_score = min(1.0, word_count / 50)  # Cap at 50 words
        score += 0.3 * length_score

        # Comparative language factor
        comparative_terms = [
            "however", "but", "whereas", "changed", "now",
            "previously", "before", "after", "instead", "rather"
        ]
        response_lower = response.lower()
        comparative_count = sum(1 for term in comparative_terms if term in response_lower)
        comparative_score = min(1.0, comparative_count / 3)
        score += 0.3 * comparative_score

        # Explanation indicators
        explanation_terms = [
            "because", "therefore", "since", "due to",
            "leads to", "results in", "explains", "shows"
        ]
        explanation_count = sum(1 for term in explanation_terms if term in response_lower)
        explanation_score = min(1.0, explanation_count / 2)
        score += 0.2 * explanation_score

        # Perspective-taking indicators
        perspective_terms = [
            "someone might", "newcomer", "novice", "expect",
            "assume", "believe", "think", "suppose"
        ]
        perspective_count = sum(1 for term in perspective_terms if term in response_lower)
        perspective_score = min(1.0, perspective_count / 2)
        score += 0.2 * perspective_score

        return np.clip(score, 0, 1)

    def _check_dual_mention(self, response: str, prompt_idx: int) -> bool:
        """Check if response mentions both old and new states."""
        # In a real implementation, this would do semantic analysis
        return len(response.split()) > 20  # Simple heuristic

    def _generate_feedback(self, mapping: Dict[str, Any]) -> str:
        """Generate adaptive feedback based on response quality."""
        quality = mapping["quality_score"]

        if quality > 0.8:
            return "Excellent! Your explanation clearly bridges the old and new understanding."
        elif quality > 0.6:
            return "Good mapping. Consider elaborating on why the change occurred."
        elif quality > 0.4:
            return "Try to explicitly contrast what was believed before vs. after the change."
        else:
            return "Focus on explaining how someone without the new information might think differently."

    def complete_session(self) -> Dict[str, Any]:
        """
        Complete the scaffolding session and compute summary metrics.

        Returns
        -------
        Dict[str, Any]
            Session summary with effectiveness metrics
        """
        if not self.active_session:
            raise RuntimeError("No active session")

        n_completed = len(self.mappings)
        n_total = len(self._generate_mapping_prompts(self.current_reversal))

        # Compute summary metrics
        completion_rate = n_completed / max(n_total, 1)
        avg_quality = np.mean([m["quality_score"] for m in self.mappings]) if self.mappings else 0
        avg_time = np.mean([m["response_time"] for m in self.mappings]) if self.mappings else 0
        dual_mention_rate = np.mean([m["mentions_both_states"] for m in self.mappings]) if self.mappings else 0

        # Estimate intervention effectiveness
        effectiveness = self._estimate_effectiveness(
            completion_rate, avg_quality, dual_mention_rate
        )

        summary = {
            "n_completed": n_completed,
            "n_total": n_total,
            "completion_rate": completion_rate,
            "avg_quality": avg_quality,
            "avg_response_time": avg_time,
            "dual_mention_rate": dual_mention_rate,
            "estimated_effectiveness": effectiveness,
            "mappings": self.mappings,
        }

        self.session_history.append(summary)
        self.active_session = False

        return summary

    def _estimate_effectiveness(
        self,
        completion_rate: float,
        avg_quality: float,
        dual_mention_rate: float
    ) -> float:
        """Estimate intervention effectiveness from session metrics."""
        # Weighted combination
        effectiveness = (
            0.3 * completion_rate +
            0.4 * avg_quality +
            0.3 * dual_mention_rate
        )
        return effectiveness * self.config.intensity


class InterventionSimulator:
    """
    Simulator for predicting intervention effects on the reversal curse.

    Uses the Bayesian curse model to simulate how different interventions
    would affect curse magnitude, allowing comparison of intervention
    strategies without running full experiments.
    """

    def __init__(self, curse_model: Optional[BayesianCurseModel] = None):
        """
        Initialize the simulator.

        Parameters
        ----------
        curse_model : BayesianCurseModel, optional
            Base curse model for predictions
        """
        self.curse_model = curse_model or BayesianCurseModel()
        self.simulation_history: List[Dict[str, Any]] = []

    def simulate_intervention(
        self,
        reversal: ReversalEvent,
        intervention_type: InterventionType,
        intensity: float = 0.7,
        n_simulations: int = 1000
    ) -> InterventionResult:
        """
        Simulate the effect of an intervention.

        Parameters
        ----------
        reversal : ReversalEvent
            The reversal event
        intervention_type : InterventionType
            Type of intervention to simulate
        intensity : float
            Intervention intensity (0-1)
        n_simulations : int
            Number of Monte Carlo simulations

        Returns
        -------
        InterventionResult
            Simulated intervention results
        """
        # Get baseline curse
        baseline, components = self.curse_model.predict_curse_magnitude(
            reversal, return_components=True
        )

        # Get intervention effects on each component
        effects = self._get_intervention_effects(intervention_type, intensity)

        # Simulate post-intervention curse
        simulated_curses = []

        for _ in range(n_simulations):
            # Apply intervention effects with noise
            modified_components = {}
            for comp, value in components.items():
                effect = effects.get(comp, 0)
                noise = np.random.normal(0, 0.05)
                modified_components[comp] = max(0, value * (1 - effect) + noise)

            # Recalculate curse
            post_curse = (
                self.curse_model.params_["load_sensitivity"] * modified_components["cognitive_load"] +
                0.3 * modified_components["updating_cost"] +
                self.curse_model.params_["temporal_weight"] * modified_components["temporal_penalty"] +
                0.3 * modified_components["tom_degradation"]
            )
            post_curse = np.clip(post_curse, 0, 1)
            simulated_curses.append(post_curse)

        simulated_curses = np.array(simulated_curses)
        mean_post = simulated_curses.mean()

        # Compute statistics
        reduction = baseline - mean_post
        reduction_percent = (reduction / baseline * 100) if baseline > 0 else 0

        # Confidence interval
        ci_low = np.percentile(simulated_curses, 2.5)
        ci_high = np.percentile(simulated_curses, 97.5)

        # Estimate engagement based on intervention type
        engagement = self._estimate_engagement(intervention_type, intensity)

        result = InterventionResult(
            baseline_curse=baseline,
            post_intervention_curse=mean_post,
            reduction_magnitude=reduction,
            reduction_percent=reduction_percent,
            confidence_interval=(baseline - ci_high, baseline - ci_low),
            component_effects=effects,
            participant_engagement=engagement,
            time_spent_seconds=int(intensity * 600)  # Up to 10 minutes
        )

        self.simulation_history.append({
            "reversal": reversal,
            "intervention_type": intervention_type.value,
            "result": result
        })

        return result

    def _get_intervention_effects(
        self,
        intervention_type: InterventionType,
        intensity: float
    ) -> Dict[str, float]:
        """Get the effect of an intervention on each curse component."""
        # Base effects by intervention type (as reduction factors)
        base_effects = {
            InterventionType.EXPLICIT_MAPPING: {
                "cognitive_load": 0.4,
                "updating_cost": 0.2,
                "temporal_penalty": 0.1,
                "tom_degradation": 0.5,
            },
            InterventionType.PERSPECTIVE_TAKING: {
                "cognitive_load": 0.2,
                "updating_cost": 0.1,
                "temporal_penalty": 0.1,
                "tom_degradation": 0.7,
            },
            InterventionType.UNCERTAINTY_HIGHLIGHTING: {
                "cognitive_load": 0.3,
                "updating_cost": 0.4,
                "temporal_penalty": 0.1,
                "tom_degradation": 0.4,
            },
            InterventionType.TEMPORAL_REFRAMING: {
                "cognitive_load": 0.2,
                "updating_cost": 0.2,
                "temporal_penalty": 0.6,
                "tom_degradation": 0.3,
            },
            InterventionType.STRUCTURAL_VISUALIZATION: {
                "cognitive_load": 0.5,
                "updating_cost": 0.3,
                "temporal_penalty": 0.2,
                "tom_degradation": 0.4,
            },
            InterventionType.INCREMENTAL_BRIDGING: {
                "cognitive_load": 0.6,
                "updating_cost": 0.5,
                "temporal_penalty": 0.3,
                "tom_degradation": 0.5,
            },
        }

        effects = base_effects.get(intervention_type, {})

        # Scale by intensity
        scaled_effects = {k: v * intensity for k, v in effects.items()}

        return scaled_effects

    def _estimate_engagement(
        self,
        intervention_type: InterventionType,
        intensity: float
    ) -> float:
        """Estimate participant engagement level."""
        # Base engagement by type (some interventions are more engaging)
        base_engagement = {
            InterventionType.EXPLICIT_MAPPING: 0.75,
            InterventionType.PERSPECTIVE_TAKING: 0.80,
            InterventionType.UNCERTAINTY_HIGHLIGHTING: 0.60,
            InterventionType.TEMPORAL_REFRAMING: 0.65,
            InterventionType.STRUCTURAL_VISUALIZATION: 0.85,
            InterventionType.INCREMENTAL_BRIDGING: 0.70,
        }

        base = base_engagement.get(intervention_type, 0.70)

        # Higher intensity can decrease engagement (fatigue)
        fatigue_factor = 1 - 0.2 * intensity

        return base * fatigue_factor

    def compare_interventions(
        self,
        reversal: ReversalEvent,
        intensities: Optional[List[float]] = None
    ) -> Dict[str, Dict[str, InterventionResult]]:
        """
        Compare all intervention types.

        Parameters
        ----------
        reversal : ReversalEvent
            The reversal event
        intensities : List[float], optional
            Intensity levels to test

        Returns
        -------
        Dict
            Results for each intervention type and intensity
        """
        if intensities is None:
            intensities = [0.3, 0.5, 0.7, 0.9]

        results = {}

        for intervention_type in InterventionType:
            results[intervention_type.value] = {}

            for intensity in intensities:
                result = self.simulate_intervention(
                    reversal, intervention_type, intensity
                )
                results[intervention_type.value][f"intensity_{intensity}"] = result

        return results

    def find_optimal_intervention(
        self,
        reversal: ReversalEvent,
        constraint: str = "reduction",
        max_time_seconds: int = 600
    ) -> Tuple[InterventionType, float, InterventionResult]:
        """
        Find the optimal intervention for a given reversal.

        Parameters
        ----------
        reversal : ReversalEvent
            The reversal event
        constraint : str
            Optimization target: "reduction", "efficiency", "engagement"
        max_time_seconds : int
            Maximum time budget

        Returns
        -------
        Tuple
            Best intervention type, intensity, and result
        """
        best_score = float('-inf')
        best_type = None
        best_intensity = None
        best_result = None

        for intervention_type in InterventionType:
            for intensity in np.linspace(0.1, 1.0, 10):
                result = self.simulate_intervention(
                    reversal, intervention_type, intensity, n_simulations=100
                )

                # Check time constraint
                if result.time_spent_seconds > max_time_seconds:
                    continue

                # Compute score based on constraint
                if constraint == "reduction":
                    score = result.reduction_magnitude
                elif constraint == "efficiency":
                    # Reduction per unit time
                    score = result.reduction_magnitude / (result.time_spent_seconds / 60)
                elif constraint == "engagement":
                    # Weighted combination of reduction and engagement
                    score = 0.6 * result.reduction_magnitude + 0.4 * result.participant_engagement
                else:
                    score = result.reduction_magnitude

                if score > best_score:
                    best_score = score
                    best_type = intervention_type
                    best_intensity = intensity
                    best_result = result

        return best_type, best_intensity, best_result


def compute_debiasing_effect(
    reversal: ReversalEvent,
    intervention_type: str = "explicit_mapping",
    intensity: float = 0.7
) -> Dict[str, Any]:
    """
    Convenience function to compute expected debiasing effect.

    Parameters
    ----------
    reversal : ReversalEvent
        The reversal event
    intervention_type : str
        Type of intervention
    intensity : float
        Intervention intensity

    Returns
    -------
    Dict[str, Any]
        Debiasing effect summary
    """
    # Map string to enum
    type_map = {t.value: t for t in InterventionType}
    itype = type_map.get(intervention_type, InterventionType.EXPLICIT_MAPPING)

    simulator = InterventionSimulator()
    result = simulator.simulate_intervention(reversal, itype, intensity)

    return {
        "baseline_curse": result.baseline_curse,
        "post_intervention_curse": result.post_intervention_curse,
        "absolute_reduction": result.reduction_magnitude,
        "percent_reduction": result.reduction_percent,
        "confidence_interval": result.confidence_interval,
        "expected_engagement": result.participant_engagement,
        "time_required_minutes": result.time_spent_seconds / 60,
        "component_effects": result.component_effects,
    }


@dataclass
class AdaptiveInterventionProtocol:
    """
    Adaptive intervention protocol that adjusts based on participant response.

    This protocol monitors participant performance and engagement,
    adjusting intervention intensity and type in real-time.
    """

    initial_type: InterventionType = InterventionType.EXPLICIT_MAPPING
    initial_intensity: float = 0.5
    adaptation_rate: float = 0.1
    min_intensity: float = 0.2
    max_intensity: float = 0.9
    performance_threshold: float = 0.6

    current_type: InterventionType = field(init=False)
    current_intensity: float = field(init=False)
    performance_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.current_type = self.initial_type
        self.current_intensity = self.initial_intensity

    def update(self, performance_score: float, engagement_score: float) -> None:
        """
        Update intervention parameters based on performance.

        Parameters
        ----------
        performance_score : float
            How well the participant performed (0-1)
        engagement_score : float
            How engaged the participant was (0-1)
        """
        self.performance_history.append(performance_score)

        # Adjust intensity based on performance
        if performance_score < self.performance_threshold:
            # Increase intensity if performance is low
            self.current_intensity = min(
                self.max_intensity,
                self.current_intensity + self.adaptation_rate
            )
        elif performance_score > 0.8 and engagement_score < 0.5:
            # Decrease intensity if performing well but disengaged
            self.current_intensity = max(
                self.min_intensity,
                self.current_intensity - self.adaptation_rate
            )

        # Consider switching intervention type after multiple low performances
        if len(self.performance_history) >= 3:
            recent_avg = np.mean(self.performance_history[-3:])
            if recent_avg < 0.4:
                self._switch_intervention_type()

    def _switch_intervention_type(self) -> None:
        """Switch to a different intervention type."""
        types = list(InterventionType)
        current_idx = types.index(self.current_type)
        # Cycle to next type
        self.current_type = types[(current_idx + 1) % len(types)]
        logger.info(f"Switching intervention type to {self.current_type.value}")

    def get_current_config(self) -> InterventionConfig:
        """Get current intervention configuration."""
        return InterventionConfig(
            intensity=self.current_intensity,
            duration_minutes=int(self.current_intensity * 15),  # 3-15 minutes
            n_mapping_prompts=max(3, int(self.current_intensity * 8)),
            include_reflection=self.current_intensity > 0.5,
            feedback_mode="adaptive",
            personalized=True
        )


class GroupInterventionDesigner:
    """
    Design interventions optimized for groups with varying knowledge states.

    When an expert needs to communicate with a group of novices who have
    different baseline knowledge, this class helps design interventions
    that address the full range of the "curse" effects.
    """

    def __init__(self):
        self.simulator = InterventionSimulator()

    def design_group_intervention(
        self,
        expert_reversal: ReversalEvent,
        novice_knowledge_levels: List[float],
        time_budget_minutes: int = 20
    ) -> Dict[str, Any]:
        """
        Design an intervention for a group communication scenario.

        Parameters
        ----------
        expert_reversal : ReversalEvent
            The expert's knowledge reversal
        novice_knowledge_levels : List[float]
            Knowledge levels of novices (0 = complete novice, 1 = near-expert)
        time_budget_minutes : int
            Total time available

        Returns
        -------
        Dict[str, Any]
            Intervention design with staged components
        """
        n_novices = len(novice_knowledge_levels)
        mean_knowledge = np.mean(novice_knowledge_levels)
        knowledge_variance = np.var(novice_knowledge_levels)

        # Determine intervention complexity based on group heterogeneity
        if knowledge_variance > 0.1:
            # Heterogeneous group: need differentiated approach
            intervention_approach = "differentiated"
        else:
            # Homogeneous group: single intervention
            intervention_approach = "unified"

        # Find optimal base intervention
        best_type, best_intensity, best_result = self.simulator.find_optimal_intervention(
            expert_reversal,
            constraint="efficiency",
            max_time_seconds=time_budget_minutes * 60
        )

        # Adjust for group
        stages = []

        if intervention_approach == "differentiated":
            # Stage 1: Common foundation
            stages.append({
                "name": "Foundation",
                "type": InterventionType.EXPLICIT_MAPPING.value,
                "intensity": 0.5,
                "duration_minutes": int(time_budget_minutes * 0.4),
                "target_group": "all",
                "description": "Establish common ground by mapping key conceptual changes"
            })

            # Stage 2: Targeted support
            stages.append({
                "name": "Targeted Support",
                "type": InterventionType.PERSPECTIVE_TAKING.value,
                "intensity": 0.7,
                "duration_minutes": int(time_budget_minutes * 0.3),
                "target_group": "low_knowledge",
                "description": "Additional scaffolding for novices with less background"
            })

            # Stage 3: Integration
            stages.append({
                "name": "Integration",
                "type": InterventionType.STRUCTURAL_VISUALIZATION.value,
                "intensity": 0.6,
                "duration_minutes": int(time_budget_minutes * 0.3),
                "target_group": "all",
                "description": "Integrate new and old understanding visually"
            })
        else:
            # Single unified intervention
            stages.append({
                "name": "Unified Intervention",
                "type": best_type.value,
                "intensity": best_intensity,
                "duration_minutes": time_budget_minutes,
                "target_group": "all",
                "description": f"Optimal intervention for homogeneous group"
            })

        # Estimate overall effectiveness
        total_reduction = sum(
            best_result.reduction_magnitude * (s["duration_minutes"] / time_budget_minutes)
            for s in stages
        )

        return {
            "approach": intervention_approach,
            "stages": stages,
            "estimated_curse_reduction": total_reduction,
            "group_stats": {
                "n_novices": n_novices,
                "mean_knowledge_level": mean_knowledge,
                "knowledge_variance": knowledge_variance
            },
            "time_budget_minutes": time_budget_minutes,
            "optimal_base_intervention": best_type.value
        }
