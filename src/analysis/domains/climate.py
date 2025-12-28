"""
Climate Science Domain Module for Reversal Curse Research.

This module provides specialized frameworks for studying the reversal curse
in climate science communication, particularly around projection updates.

Key research questions:
1. How do climate scientists communicate updated projections to the public?
2. Does the reversal curse affect public understanding of climate risk?
3. What interventions improve communication of scientific updates?

Real-world examples studied:
- Sea level rise projection updates
- Arctic ice extent revisions
- Emission pathway scenario changes
- Extreme weather attribution updates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ProjectionUpdateType(Enum):
    """Types of climate projection updates."""

    ACCELERATION = "acceleration"  # Faster than expected
    DECELERATION = "deceleration"  # Slower than expected
    DIRECTION_CHANGE = "direction_change"  # Trend reversal
    THRESHOLD_REVISION = "threshold_revision"  # Tipping point changed
    UNCERTAINTY_EXPANSION = "uncertainty_expansion"  # Wider confidence intervals
    UNCERTAINTY_REDUCTION = "uncertainty_reduction"  # Narrower confidence intervals
    MECHANISM_DISCOVERY = "mechanism_discovery"  # New causal pathway identified


class CommunicationChannel(Enum):
    """Channels for climate communication."""

    IPCC_REPORT = "ipcc_report"
    PEER_REVIEWED_PAPER = "peer_reviewed_paper"
    PRESS_RELEASE = "press_release"
    MEDIA_INTERVIEW = "media_interview"
    SOCIAL_MEDIA = "social_media"
    PUBLIC_LECTURE = "public_lecture"
    POLICY_BRIEFING = "policy_briefing"


class AudienceType(Enum):
    """Types of audience for climate communication."""

    GENERAL_PUBLIC = "general_public"
    POLICYMAKERS = "policymakers"
    JOURNALISTS = "journalists"
    EDUCATORS = "educators"
    INDUSTRY = "industry"
    ACTIVISTS = "activists"


@dataclass
class ProjectionUpdate:
    """
    Representation of a climate projection update.

    Captures the nature and context of scientific knowledge updates
    that may trigger reversal curse effects.
    """

    update_id: str
    domain: str  # e.g., "sea_level", "temperature", "precipitation"
    phenomenon: str  # Specific phenomenon being projected
    update_type: ProjectionUpdateType
    update_date: datetime

    # Pre-update projection
    previous_central_estimate: float
    previous_uncertainty_range: Tuple[float, float]
    previous_time_horizon_year: int
    previous_source: str  # e.g., "IPCC AR5"
    previous_duration_years: float  # How long old projection was consensus

    # Post-update projection
    new_central_estimate: float
    new_uncertainty_range: Tuple[float, float]
    new_time_horizon_year: int
    new_source: str

    # Context
    triggering_evidence: str
    media_coverage_level: str
    policy_relevance: str  # "low", "moderate", "high", "critical"

    @property
    def relative_change(self) -> float:
        """Compute relative change in central estimate."""
        if self.previous_central_estimate == 0:
            return np.inf if self.new_central_estimate != 0 else 0
        return (
            (self.new_central_estimate - self.previous_central_estimate) /
            abs(self.previous_central_estimate)
        )

    @property
    def direction_consistent(self) -> bool:
        """Check if update is in same direction (worse/better)."""
        # For climate impacts, larger magnitude is generally "worse"
        return np.sign(self.new_central_estimate) == np.sign(self.previous_central_estimate)

    @property
    def expected_curse_severity(self) -> float:
        """
        Estimate expected reversal curse severity.

        Higher for:
        - Counter-intuitive direction changes
        - Longer-standing projections
        - Higher policy relevance
        - More media coverage
        """
        # Direction change increases curse
        direction_factor = 0.0 if self.direction_consistent else 0.3

        # Large relative changes increase curse
        change_factor = min(1.0, abs(self.relative_change))

        # Duration of old projection
        duration_factor = min(1.0, self.previous_duration_years / 15)

        # Policy relevance
        policy_weights = {
            "low": 0.2,
            "moderate": 0.4,
            "high": 0.7,
            "critical": 1.0
        }
        policy_factor = policy_weights.get(self.policy_relevance, 0.5)

        severity = (
            0.3 * direction_factor +
            0.25 * change_factor +
            0.25 * duration_factor +
            0.2 * policy_factor
        )

        return severity


@dataclass
class ScientistCommunicationRecord:
    """Record of how a scientist communicated a projection update."""

    scientist_id: str
    update_id: str
    channel: CommunicationChannel
    audience: AudienceType
    timestamp: datetime

    # Communication characteristics
    mentioned_previous_projection: bool
    explained_reason_for_update: bool
    discussed_uncertainty: bool
    used_visualizations: bool
    acknowledged_public_confusion: bool
    provided_context_for_change: bool
    emphasized_consistent_aspects: bool

    # Reception metrics (where available)
    audience_size: Optional[int] = None
    engagement_score: Optional[float] = None  # 0-1
    media_pickup_count: Optional[int] = None


@dataclass
class PublicUnderstandingRecord:
    """Record of public understanding after projection update."""

    respondent_id: str
    update_id: str
    survey_date: datetime
    days_since_update: int

    # Demographics
    age_group: str
    education_level: str
    political_orientation: float  # -1 (left) to 1 (right)
    prior_climate_concern: float  # 0-10

    # Knowledge outcomes
    knows_projections_updated: bool
    correctly_states_new_projection: bool
    correctly_states_direction_of_change: bool
    confuses_old_new: bool  # KEY CURSE INDICATOR

    # Attitude outcomes
    perceived_scientist_credibility: float  # 0-10
    climate_concern_level: float  # 0-10
    policy_support_level: float  # 0-10
    personal_action_intent: float  # 0-10


class ClimateReversalAnalyzer:
    """
    Analyzer for climate communication reversal curse phenomena.

    Studies how climate projection updates affect scientist-public
    communication and public understanding.
    """

    def __init__(self):
        self.projection_updates: Dict[str, ProjectionUpdate] = {}
        self.communication_records: List[ScientistCommunicationRecord] = []
        self.public_understanding: List[PublicUnderstandingRecord] = []

    def register_projection_update(self, update: ProjectionUpdate) -> None:
        """Register a projection update for tracking."""
        self.projection_updates[update.update_id] = update

    def record_communication(
        self,
        record: ScientistCommunicationRecord
    ) -> None:
        """Record a scientist communication event."""
        self.communication_records.append(record)

    def record_public_understanding(
        self,
        record: PublicUnderstandingRecord
    ) -> None:
        """Record a public understanding measurement."""
        self.public_understanding.append(record)

    def analyze_curse_in_communication(
        self,
        update_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze reversal curse effects in scientist communication.

        Parameters
        ----------
        update_id : str, optional
            Filter to specific projection update

        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        records = self.communication_records
        if update_id:
            records = [r for r in records if r.update_id == update_id]

        if not records:
            return {"error": "No records found"}

        df = pd.DataFrame([
            {
                "scientist_id": r.scientist_id,
                "channel": r.channel.value,
                "audience": r.audience.value,
                "mentioned_previous": r.mentioned_previous_projection,
                "explained_reason": r.explained_reason_for_update,
                "discussed_uncertainty": r.discussed_uncertainty,
                "used_visualizations": r.used_visualizations,
                "acknowledged_confusion": r.acknowledged_public_confusion,
                "emphasized_consistent": r.emphasized_consistent_aspects,
            }
            for r in records
        ])

        results = {
            "n_communications": len(df),
            "by_channel": df.groupby("channel").size().to_dict(),
            "by_audience": df.groupby("audience").size().to_dict(),
            "communication_practices": {
                "mentioned_previous_rate": df["mentioned_previous"].mean(),
                "explained_reason_rate": df["explained_reason"].mean(),
                "discussed_uncertainty_rate": df["discussed_uncertainty"].mean(),
                "used_visualizations_rate": df["used_visualizations"].mean(),
                "acknowledged_confusion_rate": df["acknowledged_confusion"].mean(),
                "emphasized_consistent_rate": df["emphasized_consistent"].mean(),
            }
        }

        # Compare channels
        channel_effectiveness = {}
        for channel in df["channel"].unique():
            channel_df = df[df["channel"] == channel]
            channel_effectiveness[channel] = {
                "n": len(channel_df),
                "complete_communication_rate": (
                    channel_df["mentioned_previous"] &
                    channel_df["explained_reason"] &
                    channel_df["discussed_uncertainty"]
                ).mean()
            }
        results["channel_effectiveness"] = channel_effectiveness

        return results

    def analyze_public_understanding(
        self,
        update_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze public understanding following projection updates.

        Parameters
        ----------
        update_id : str, optional
            Filter to specific update

        Returns
        -------
        Dict[str, Any]
            Understanding analysis
        """
        records = self.public_understanding
        if update_id:
            records = [r for r in records if r.update_id == update_id]

        if not records:
            return {"error": "No records found"}

        df = pd.DataFrame([
            {
                "respondent_id": r.respondent_id,
                "days_since_update": r.days_since_update,
                "age_group": r.age_group,
                "education": r.education_level,
                "political_orientation": r.political_orientation,
                "prior_concern": r.prior_climate_concern,
                "knows_updated": r.knows_projections_updated,
                "correct_projection": r.correctly_states_new_projection,
                "correct_direction": r.correctly_states_direction_of_change,
                "confusion": r.confuses_old_new,  # KEY METRIC
                "credibility": r.perceived_scientist_credibility,
                "concern": r.climate_concern_level,
                "policy_support": r.policy_support_level,
                "action_intent": r.personal_action_intent
            }
            for r in records
        ])

        # Knowledge metrics
        knowledge = {
            "awareness_rate": df["knows_updated"].mean(),
            "correct_projection_rate": df["correct_projection"].mean(),
            "correct_direction_rate": df["correct_direction"].mean(),
            "confusion_rate": df["confusion"].mean(),  # KEY CURSE METRIC
        }

        # Attitude metrics
        attitudes = {
            "mean_credibility": df["credibility"].mean(),
            "mean_concern": df["concern"].mean(),
            "mean_policy_support": df["policy_support"].mean(),
            "mean_action_intent": df["action_intent"].mean(),
        }

        # Temporal patterns in confusion
        temporal = df.groupby(
            pd.cut(df["days_since_update"], bins=[0, 7, 30, 90, 365])
        ).agg({
            "confusion": "mean",
            "correct_projection": "mean"
        }).to_dict()

        # Demographic predictors of confusion
        demographic_effects = {}

        # Education effect
        if len(df["education"].unique()) > 1:
            edu_groups = df.groupby("education")["confusion"].mean()
            demographic_effects["education"] = edu_groups.to_dict()

        # Political orientation effect
        if len(df) > 20:
            r, p = stats.pearsonr(df["political_orientation"], df["confusion"])
            demographic_effects["political_orientation"] = {
                "correlation_with_confusion": r,
                "p_value": p
            }

        # Prior concern effect
        if len(df) > 20:
            r, p = stats.pearsonr(df["prior_concern"], df["confusion"])
            demographic_effects["prior_concern"] = {
                "correlation_with_confusion": r,
                "p_value": p
            }

        return {
            "n_respondents": len(df),
            "knowledge_metrics": knowledge,
            "attitude_metrics": attitudes,
            "temporal_patterns": temporal,
            "demographic_effects": demographic_effects
        }

    def analyze_credibility_impact(self) -> Dict[str, Any]:
        """
        Analyze how projection updates affect scientist credibility.

        This is a key concern: does updating projections undermine trust?

        Returns
        -------
        Dict[str, Any]
            Credibility analysis
        """
        if not self.public_understanding:
            return {"error": "No public understanding data"}

        df = pd.DataFrame([
            {
                "update_id": r.update_id,
                "confusion": r.confuses_old_new,
                "credibility": r.perceived_scientist_credibility,
                "prior_concern": r.prior_climate_concern,
                "correct_direction": r.correctly_states_direction_of_change
            }
            for r in self.public_understanding
        ])

        # Overall credibility
        mean_credibility = df["credibility"].mean()

        # Credibility by confusion status
        confused_credibility = df[df["confusion"]]["credibility"].mean()
        not_confused_credibility = df[~df["confusion"]]["credibility"].mean()

        # Statistical test
        if df["confusion"].sum() > 5 and (~df["confusion"]).sum() > 5:
            t_stat, p_value = stats.ttest_ind(
                df[df["confusion"]]["credibility"],
                df[~df["confusion"]]["credibility"]
            )
            effect_size = (not_confused_credibility - confused_credibility) / df["credibility"].std()
        else:
            t_stat = p_value = effect_size = np.nan

        # Credibility by update
        by_update = {}
        for uid, update in self.projection_updates.items():
            update_df = df[df["update_id"] == uid]
            if len(update_df) > 0:
                by_update[uid] = {
                    "mean_credibility": update_df["credibility"].mean(),
                    "confusion_rate": update_df["confusion"].mean(),
                    "expected_severity": update.expected_curse_severity
                }

        # Correlation: expected severity vs actual credibility impact
        if len(by_update) >= 3:
            severities = [v["expected_severity"] for v in by_update.values()]
            credibilities = [v["mean_credibility"] for v in by_update.values()]
            severity_credibility_r, severity_credibility_p = stats.pearsonr(
                severities, credibilities
            )
        else:
            severity_credibility_r = severity_credibility_p = np.nan

        return {
            "overall_mean_credibility": mean_credibility,
            "credibility_by_confusion": {
                "confused": confused_credibility,
                "not_confused": not_confused_credibility,
                "difference": not_confused_credibility - confused_credibility,
                "t_statistic": t_stat,
                "p_value": p_value,
                "cohens_d": effect_size
            },
            "by_update": by_update,
            "severity_credibility_relationship": {
                "correlation": severity_credibility_r,
                "p_value": severity_credibility_p
            }
        }


@dataclass
class PublicCommunicationStudy:
    """
    Framework for studying public communication after projection updates.

    Provides structured methodology for measuring reversal curse effects
    in climate science communication.
    """

    study_name: str
    projection_update: ProjectionUpdate
    target_sample_size: int = 1000
    survey_waves: List[int] = field(
        default_factory=lambda: [7, 30, 90, 180]
    )  # Days after update
    include_control: bool = True  # Include respondents not exposed to update

    def generate_survey_instrument(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate survey instrument for public understanding study.

        Returns
        -------
        Dict[str, List[Dict]]
            Survey sections with items
        """
        update = self.projection_update

        survey = {
            "awareness": [
                {
                    "id": "a1",
                    "question": f"Have you heard that scientists updated their projections for {update.phenomenon}?",
                    "type": "yes_no_unsure"
                },
                {
                    "id": "a2",
                    "question": "How did you learn about this update?",
                    "type": "multiple_choice",
                    "options": [
                        "News media", "Social media", "Scientific publication",
                        "Government agency", "Friend/family", "Other", "Did not hear about it"
                    ]
                }
            ],
            "knowledge_current": [
                {
                    "id": "k1",
                    "question": f"According to current scientific projections, by {update.new_time_horizon_year}, "
                                f"{update.phenomenon} will be approximately:",
                    "type": "numeric_estimate",
                    "correct_answer": update.new_central_estimate,
                    "unit": self._get_unit(update.domain)
                },
                {
                    "id": "k2",
                    "question": f"Compared to previous projections from {update.previous_source}, "
                                "current projections show the impact will be:",
                    "type": "multiple_choice",
                    "options": ["Much larger", "Somewhat larger", "About the same",
                               "Somewhat smaller", "Much smaller", "Don't know"],
                    "correct_answer": self._get_direction_answer(update)
                }
            ],
            "knowledge_previous": [
                {
                    "id": "kp1",
                    "question": f"What did {update.previous_source} project for {update.phenomenon}?",
                    "type": "numeric_estimate",
                    "correct_answer": update.previous_central_estimate
                }
            ],
            "confusion_check": [
                {
                    "id": "c1",
                    "question": "I feel confident I understand the current scientific consensus on this topic.",
                    "type": "likert_7"
                },
                {
                    "id": "c2",
                    "question": "It's difficult to keep track of what scientists currently project vs. what they used to project.",
                    "type": "likert_7"
                },
                {
                    "id": "c3",
                    "question": "I sometimes mix up older and newer scientific projections.",
                    "type": "likert_7"
                }
            ],
            "credibility": [
                {
                    "id": "cr1",
                    "question": "Climate scientists are trustworthy sources of information.",
                    "type": "likert_7"
                },
                {
                    "id": "cr2",
                    "question": "When scientists update their projections, it shows the scientific process is working.",
                    "type": "likert_7"
                },
                {
                    "id": "cr3",
                    "question": "Changes in scientific projections make me less confident in science.",
                    "type": "likert_7"
                }
            ],
            "concern_and_action": [
                {
                    "id": "ca1",
                    "question": "How concerned are you about climate change?",
                    "type": "scale_0_10"
                },
                {
                    "id": "ca2",
                    "question": "How likely are you to support climate policies?",
                    "type": "scale_0_10"
                },
                {
                    "id": "ca3",
                    "question": "How likely are you to change your personal behavior to address climate change?",
                    "type": "scale_0_10"
                }
            ],
            "demographics": [
                {
                    "id": "d1",
                    "question": "What is your age?",
                    "type": "age_range"
                },
                {
                    "id": "d2",
                    "question": "What is your highest level of education?",
                    "type": "education_level"
                },
                {
                    "id": "d3",
                    "question": "In politics, where would you place yourself on a scale from very liberal (1) to very conservative (7)?",
                    "type": "scale_1_7"
                }
            ]
        }

        return survey

    def _get_unit(self, domain: str) -> str:
        """Get appropriate unit for domain."""
        units = {
            "sea_level": "meters",
            "temperature": "°C",
            "precipitation": "% change",
            "ice_extent": "million km²",
            "emissions": "GtCO2"
        }
        return units.get(domain, "units")

    def _get_direction_answer(self, update: ProjectionUpdate) -> str:
        """Determine correct answer for direction question."""
        change = update.relative_change
        if change > 0.2:
            return "Much larger"
        elif change > 0.05:
            return "Somewhat larger"
        elif change > -0.05:
            return "About the same"
        elif change > -0.2:
            return "Somewhat smaller"
        else:
            return "Much smaller"

    def calculate_sample_size(
        self,
        expected_effect: float = 0.1,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> Dict[str, int]:
        """
        Calculate required sample size for study.

        Parameters
        ----------
        expected_effect : float
            Expected difference in confusion rate between conditions
        alpha : float
            Significance level
        power : float
            Statistical power

        Returns
        -------
        Dict[str, int]
            Sample size requirements
        """
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        # Baseline confusion rate estimate
        p1 = 0.35
        p2 = p1 - expected_effect

        pooled_p = (p1 + p2) / 2
        n_per_group = (
            2 * ((z_alpha + z_beta) ** 2) * pooled_p * (1 - pooled_p)
        ) / (expected_effect ** 2)

        n_per_group = int(np.ceil(n_per_group))

        # Account for multiple waves (with attrition)
        attrition_rate = 0.15  # Per wave
        n_adjusted = int(n_per_group / ((1 - attrition_rate) ** (len(self.survey_waves) - 1)))

        return {
            "n_per_wave_baseline": n_per_group,
            "n_initial_accounting_attrition": n_adjusted,
            "total_with_control": n_adjusted * 2 if self.include_control else n_adjusted,
            "expected_final_n": int(n_adjusted * (1 - attrition_rate) ** (len(self.survey_waves) - 1)),
            "assumptions": {
                "baseline_confusion_rate": p1,
                "expected_reduction": expected_effect,
                "attrition_rate_per_wave": attrition_rate,
                "n_waves": len(self.survey_waves)
            }
        }


# Pre-defined notable climate projection updates
NOTABLE_CLIMATE_UPDATES = [
    ProjectionUpdate(
        update_id="SEA_LEVEL_2021",
        domain="sea_level",
        phenomenon="Global mean sea level rise by 2100",
        update_type=ProjectionUpdateType.ACCELERATION,
        update_date=datetime(2021, 8, 1),
        previous_central_estimate=0.63,  # meters
        previous_uncertainty_range=(0.29, 1.1),
        previous_time_horizon_year=2100,
        previous_source="IPCC AR5",
        previous_duration_years=8,
        new_central_estimate=0.77,
        new_uncertainty_range=(0.38, 1.01),
        new_time_horizon_year=2100,
        new_source="IPCC AR6",
        triggering_evidence="Improved ice sheet models, observations",
        media_coverage_level="high",
        policy_relevance="critical"
    ),
    ProjectionUpdate(
        update_id="ARCTIC_ICE_2020",
        domain="ice_extent",
        phenomenon="Arctic summer ice-free date",
        update_type=ProjectionUpdateType.ACCELERATION,
        update_date=datetime(2020, 9, 1),
        previous_central_estimate=2050,  # Year
        previous_uncertainty_range=(2040, 2060),
        previous_time_horizon_year=2100,
        previous_source="IPCC AR5",
        previous_duration_years=7,
        new_central_estimate=2035,
        new_uncertainty_range=(2030, 2050),
        new_time_horizon_year=2100,
        new_source="Multiple studies",
        triggering_evidence="Faster-than-projected ice loss observed",
        media_coverage_level="high",
        policy_relevance="high"
    ),
    ProjectionUpdate(
        update_id="CARBON_BUDGET_2018",
        domain="emissions",
        phenomenon="Remaining carbon budget for 1.5°C",
        update_type=ProjectionUpdateType.UNCERTAINTY_EXPANSION,
        update_date=datetime(2018, 10, 1),
        previous_central_estimate=400,  # GtCO2
        previous_uncertainty_range=(200, 600),
        previous_time_horizon_year=2100,
        previous_source="IPCC AR5",
        previous_duration_years=5,
        new_central_estimate=580,
        new_uncertainty_range=(420, 770),
        new_time_horizon_year=2100,
        new_source="IPCC SR15",
        triggering_evidence="Updated climate sensitivity estimates",
        media_coverage_level="moderate",
        policy_relevance="critical"
    ),
    ProjectionUpdate(
        update_id="ATTRIBUTION_UPDATE_2021",
        domain="extreme_weather",
        phenomenon="Human attribution of extreme heat events",
        update_type=ProjectionUpdateType.MECHANISM_DISCOVERY,
        update_date=datetime(2021, 7, 1),
        previous_central_estimate=0.5,  # Probability human-caused
        previous_uncertainty_range=(0.3, 0.7),
        previous_time_horizon_year=2021,
        previous_source="Pre-2020 studies",
        previous_duration_years=10,
        new_central_estimate=0.95,  # For 2021 PNW heat dome
        new_uncertainty_range=(0.90, 0.99),
        new_time_horizon_year=2021,
        new_source="Rapid attribution study",
        triggering_evidence="Pacific Northwest heat dome event",
        media_coverage_level="viral",
        policy_relevance="high"
    )
]
