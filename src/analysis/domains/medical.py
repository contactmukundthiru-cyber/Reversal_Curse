"""
Medical Domain Module for Reversal Curse Research.

This module provides specialized frameworks for studying the reversal curse
in medical contexts, particularly around treatment guideline changes.

Key research questions:
1. How do physicians communicate guideline changes to patients?
2. Does the reversal curse affect shared decision-making?
3. What interventions improve post-reversal patient education?

Real-world examples studied:
- Hormone replacement therapy (HRT) recommendations
- Aspirin for primary prevention guidelines
- Dietary fat/cholesterol recommendations
- Cancer screening age thresholds
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


class GuidelineChangeType(Enum):
    """Types of medical guideline changes."""

    REVERSAL = "reversal"  # Complete reversal (was recommended -> now not, or vice versa)
    REFINEMENT = "refinement"  # Population scope changed
    THRESHOLD_SHIFT = "threshold_shift"  # Numeric threshold changed
    CONDITIONAL = "conditional"  # Now depends on factors it didn't before
    DEPRIORITIZATION = "deprioritization"  # From recommended to optional


class EvidenceLevel(Enum):
    """Evidence quality levels for guidelines."""

    HIGH = "high"  # Multiple RCTs, meta-analyses
    MODERATE = "moderate"  # Limited RCTs, observational studies
    LOW = "low"  # Expert opinion, case series
    VERY_LOW = "very_low"  # Extrapolation, biological plausibility


@dataclass
class TreatmentGuidelineChange:
    """
    Representation of a treatment guideline change.

    Captures the nature and context of medical knowledge reversals.
    """

    guideline_id: str
    condition: str
    treatment: str
    change_type: GuidelineChangeType
    change_date: datetime

    # Pre-change state
    previous_recommendation: str
    previous_evidence_level: EvidenceLevel
    previous_target_population: str
    previous_duration_years: float  # How long old guideline was in effect

    # Post-change state
    new_recommendation: str
    new_evidence_level: EvidenceLevel
    new_target_population: str

    # Context
    triggering_evidence: str  # What caused the change
    issuing_body: str  # e.g., "USPSTF", "AHA", "WHO"
    media_coverage_level: str  # "low", "moderate", "high", "viral"

    @property
    def reversal_magnitude(self) -> float:
        """Compute magnitude of the guideline change."""
        if self.change_type == GuidelineChangeType.REVERSAL:
            return 1.0
        elif self.change_type == GuidelineChangeType.THRESHOLD_SHIFT:
            return 0.6
        elif self.change_type == GuidelineChangeType.REFINEMENT:
            return 0.4
        elif self.change_type == GuidelineChangeType.CONDITIONAL:
            return 0.5
        else:
            return 0.3

    @property
    def expected_curse_severity(self) -> float:
        """
        Estimate expected reversal curse severity.

        Higher for:
        - Longer-standing guidelines
        - Higher previous evidence level
        - Higher media coverage
        """
        duration_factor = min(1.0, self.previous_duration_years / 20)

        evidence_weights = {
            EvidenceLevel.HIGH: 0.9,
            EvidenceLevel.MODERATE: 0.6,
            EvidenceLevel.LOW: 0.3,
            EvidenceLevel.VERY_LOW: 0.1
        }
        evidence_factor = evidence_weights.get(self.previous_evidence_level, 0.5)

        media_weights = {
            "low": 0.2,
            "moderate": 0.5,
            "high": 0.8,
            "viral": 1.0
        }
        media_factor = media_weights.get(self.media_coverage_level, 0.5)

        severity = (
            0.4 * self.reversal_magnitude +
            0.3 * duration_factor +
            0.2 * evidence_factor +
            0.1 * media_factor
        )

        return severity


@dataclass
class PhysicianCommunicationRecord:
    """Record of how a physician communicated a guideline change."""

    physician_id: str
    patient_id: str
    guideline_change_id: str
    timestamp: datetime

    # Communication characteristics
    mentioned_previous_guideline: bool
    explained_reason_for_change: bool
    acknowledged_confusion: bool
    provided_written_materials: bool
    time_spent_minutes: float

    # Patient outcomes
    patient_understanding_score: float  # 0-1, assessed by questionnaire
    patient_trust_change: float  # -1 to 1, change in trust
    patient_adherence_intent: float  # 0-1, intent to follow new guideline
    patient_questions_asked: int


@dataclass
class PatientOutcome:
    """Patient outcome following guideline change communication."""

    patient_id: str
    guideline_change_id: str
    follow_up_days: int

    # Behavioral outcomes
    followed_new_guideline: bool
    sought_second_opinion: bool
    changed_providers: bool
    stopped_treatment_entirely: bool

    # Knowledge outcomes
    correctly_states_new_guideline: bool
    correctly_states_reason: bool
    confuses_old_new: bool

    # Emotional outcomes
    anxiety_level: float  # 0-10
    trust_in_medicine: float  # 0-10
    health_locus_of_control: float  # internal vs external


class MedicalReversalAnalyzer:
    """
    Analyzer for medical reversal curse phenomena.

    Studies how treatment guideline changes affect physician-patient
    communication and patient outcomes.
    """

    def __init__(self):
        self.guideline_changes: Dict[str, TreatmentGuidelineChange] = {}
        self.communication_records: List[PhysicianCommunicationRecord] = []
        self.patient_outcomes: List[PatientOutcome] = []

    def register_guideline_change(
        self,
        change: TreatmentGuidelineChange
    ) -> None:
        """Register a guideline change for tracking."""
        self.guideline_changes[change.guideline_id] = change

    def record_communication(
        self,
        record: PhysicianCommunicationRecord
    ) -> None:
        """Record a physician-patient communication."""
        self.communication_records.append(record)

    def record_outcome(self, outcome: PatientOutcome) -> None:
        """Record a patient outcome."""
        self.patient_outcomes.append(outcome)

    def analyze_curse_in_communication(
        self,
        guideline_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze reversal curse effects in physician communication.

        Parameters
        ----------
        guideline_id : str, optional
            Filter to specific guideline change

        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        records = self.communication_records
        if guideline_id:
            records = [r for r in records if r.guideline_change_id == guideline_id]

        if not records:
            return {"error": "No records found"}

        df = pd.DataFrame([
            {
                "physician_id": r.physician_id,
                "mentioned_previous": r.mentioned_previous_guideline,
                "explained_reason": r.explained_reason_for_change,
                "acknowledged_confusion": r.acknowledged_confusion,
                "time_spent": r.time_spent_minutes,
                "patient_understanding": r.patient_understanding_score,
                "patient_trust_change": r.patient_trust_change,
                "adherence_intent": r.patient_adherence_intent
            }
            for r in records
        ])

        # Key metrics
        results = {
            "n_communications": len(df),
            "mentioned_previous_rate": df["mentioned_previous"].mean(),
            "explained_reason_rate": df["explained_reason"].mean(),
            "acknowledged_confusion_rate": df["acknowledged_confusion"].mean(),
            "mean_time_spent": df["time_spent"].mean(),
            "mean_patient_understanding": df["patient_understanding"].mean(),
            "mean_trust_change": df["patient_trust_change"].mean(),
            "mean_adherence_intent": df["adherence_intent"].mean(),
        }

        # Compare communication strategies
        if df["mentioned_previous"].sum() > 5 and (~df["mentioned_previous"]).sum() > 5:
            mentioned_group = df[df["mentioned_previous"]]["patient_understanding"]
            not_mentioned_group = df[~df["mentioned_previous"]]["patient_understanding"]

            t_stat, p_value = stats.ttest_ind(mentioned_group, not_mentioned_group)
            effect_size = (mentioned_group.mean() - not_mentioned_group.mean()) / df["patient_understanding"].std()

            results["mentioned_vs_not"] = {
                "understanding_with_mention": mentioned_group.mean(),
                "understanding_without_mention": not_mentioned_group.mean(),
                "t_statistic": t_stat,
                "p_value": p_value,
                "cohens_d": effect_size
            }

        return results

    def analyze_patient_outcomes(
        self,
        guideline_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze patient outcomes following guideline changes.

        Parameters
        ----------
        guideline_id : str, optional
            Filter to specific guideline change

        Returns
        -------
        Dict[str, Any]
            Outcome analysis
        """
        outcomes = self.patient_outcomes
        if guideline_id:
            outcomes = [o for o in outcomes if o.guideline_change_id == guideline_id]

        if not outcomes:
            return {"error": "No outcomes found"}

        df = pd.DataFrame([
            {
                "patient_id": o.patient_id,
                "follow_up_days": o.follow_up_days,
                "followed_guideline": o.followed_new_guideline,
                "second_opinion": o.sought_second_opinion,
                "changed_providers": o.changed_providers,
                "stopped_treatment": o.stopped_treatment_entirely,
                "correct_knowledge": o.correctly_states_new_guideline,
                "correct_reason": o.correctly_states_reason,
                "confusion": o.confuses_old_new,
                "anxiety": o.anxiety_level,
                "trust": o.trust_in_medicine
            }
            for o in outcomes
        ])

        # Behavioral outcomes
        behavioral = {
            "adherence_rate": df["followed_guideline"].mean(),
            "second_opinion_rate": df["second_opinion"].mean(),
            "provider_change_rate": df["changed_providers"].mean(),
            "treatment_abandonment_rate": df["stopped_treatment"].mean(),
        }

        # Knowledge outcomes (confusion = reversal curse indicator)
        knowledge = {
            "correct_guideline_rate": df["correct_knowledge"].mean(),
            "correct_reason_rate": df["correct_reason"].mean(),
            "confusion_rate": df["confusion"].mean(),  # KEY CURSE METRIC
        }

        # Emotional outcomes
        emotional = {
            "mean_anxiety": df["anxiety"].mean(),
            "mean_trust": df["trust"].mean(),
            "high_anxiety_rate": (df["anxiety"] > 7).mean(),
            "low_trust_rate": (df["trust"] < 5).mean(),
        }

        # Correlation: confusion with outcomes
        if len(df) > 10:
            correlations = {
                "confusion_adherence_r": stats.pearsonr(
                    df["confusion"].astype(int),
                    df["followed_guideline"].astype(int)
                )[0],
                "confusion_anxiety_r": stats.pearsonr(
                    df["confusion"].astype(int),
                    df["anxiety"]
                )[0],
                "confusion_trust_r": stats.pearsonr(
                    df["confusion"].astype(int),
                    df["trust"]
                )[0]
            }
        else:
            correlations = {}

        return {
            "n_patients": len(df),
            "behavioral_outcomes": behavioral,
            "knowledge_outcomes": knowledge,
            "emotional_outcomes": emotional,
            "confusion_correlations": correlations
        }

    def identify_risk_factors(self) -> Dict[str, Any]:
        """
        Identify factors that predict worse reversal curse effects.

        Returns
        -------
        Dict[str, Any]
            Risk factor analysis
        """
        # Merge communication and outcome data
        comm_df = pd.DataFrame([
            {
                "patient_id": r.patient_id,
                "guideline_id": r.guideline_change_id,
                "mentioned_previous": r.mentioned_previous_guideline,
                "explained_reason": r.explained_reason_for_change,
                "time_spent": r.time_spent_minutes,
                "initial_understanding": r.patient_understanding_score
            }
            for r in self.communication_records
        ])

        outcome_df = pd.DataFrame([
            {
                "patient_id": o.patient_id,
                "guideline_id": o.guideline_change_id,
                "confusion": o.confuses_old_new,
                "adherence": o.followed_new_guideline
            }
            for o in self.patient_outcomes
        ])

        if comm_df.empty or outcome_df.empty:
            return {"error": "Insufficient data"}

        merged = comm_df.merge(
            outcome_df,
            on=["patient_id", "guideline_id"],
            how="inner"
        )

        if len(merged) < 20:
            return {"error": "Insufficient matched records"}

        # Predictors of confusion
        confusion_predictors = {}

        for predictor in ["mentioned_previous", "explained_reason", "time_spent", "initial_understanding"]:
            if predictor in ["mentioned_previous", "explained_reason"]:
                # Chi-square for categorical
                crosstab = pd.crosstab(merged[predictor], merged["confusion"])
                if crosstab.shape == (2, 2):
                    chi2, p, dof, expected = stats.chi2_contingency(crosstab)
                    confusion_predictors[predictor] = {
                        "chi2": chi2,
                        "p_value": p,
                        "protective": merged[merged[predictor]]["confusion"].mean() < merged[~merged[predictor]]["confusion"].mean()
                    }
            else:
                # Point-biserial correlation for continuous
                r, p = stats.pointbiserialr(merged["confusion"], merged[predictor])
                confusion_predictors[predictor] = {
                    "correlation": r,
                    "p_value": p
                }

        # Guideline-level factors
        guideline_factors = {}
        for gid, change in self.guideline_changes.items():
            g_outcomes = [o for o in self.patient_outcomes if o.guideline_change_id == gid]
            if g_outcomes:
                confusion_rate = sum(o.confuses_old_new for o in g_outcomes) / len(g_outcomes)
                guideline_factors[gid] = {
                    "expected_severity": change.expected_curse_severity,
                    "observed_confusion_rate": confusion_rate,
                    "change_type": change.change_type.value,
                    "previous_duration_years": change.previous_duration_years
                }

        return {
            "communication_predictors": confusion_predictors,
            "guideline_factors": guideline_factors
        }


@dataclass
class PatientCommunicationStudy:
    """
    Framework for studying patient communication after guideline changes.

    Provides structured methodology for measuring reversal curse effects
    in clinical settings.
    """

    study_name: str
    guideline_change: TreatmentGuidelineChange
    target_n_physicians: int = 50
    target_n_patients_per_physician: int = 10
    follow_up_intervals_days: List[int] = field(
        default_factory=lambda: [7, 30, 90]
    )

    # Outcome measures
    primary_outcome: str = "patient_confusion_rate"
    secondary_outcomes: List[str] = field(
        default_factory=lambda: [
            "adherence_rate",
            "trust_change",
            "anxiety_level",
            "second_opinion_seeking"
        ]
    )

    def generate_patient_questionnaire(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate questionnaire for assessing patient understanding.

        Returns
        -------
        Dict[str, List[Dict]]
            Questionnaire sections with items
        """
        change = self.guideline_change

        questionnaire = {
            "knowledge_test": [
                {
                    "id": "k1",
                    "question": f"What is the current recommendation for {change.treatment}?",
                    "correct_answer": change.new_recommendation,
                    "type": "free_response"
                },
                {
                    "id": "k2",
                    "question": f"What was the previous recommendation for {change.treatment}?",
                    "correct_answer": change.previous_recommendation,
                    "type": "free_response"
                },
                {
                    "id": "k3",
                    "question": "Why did the recommendation change?",
                    "correct_answer": change.triggering_evidence,
                    "type": "free_response"
                },
                {
                    "id": "k4",
                    "question": f"The current guideline recommends {change.new_recommendation}. True or False?",
                    "correct_answer": "True",
                    "type": "true_false"
                },
                {
                    "id": "k5",
                    "question": f"The guideline about {change.treatment} has never changed. True or False?",
                    "correct_answer": "False",
                    "type": "true_false"
                }
            ],
            "confusion_indicators": [
                {
                    "id": "c1",
                    "question": "I feel confident about what I should do regarding this treatment.",
                    "type": "likert_5"
                },
                {
                    "id": "c2",
                    "question": "I sometimes mix up the old and new recommendations.",
                    "type": "likert_5"
                },
                {
                    "id": "c3",
                    "question": "It's hard to keep track of what doctors recommend now vs. before.",
                    "type": "likert_5"
                }
            ],
            "trust_measures": [
                {
                    "id": "t1",
                    "question": "I trust that medical recommendations are based on good evidence.",
                    "type": "likert_7"
                },
                {
                    "id": "t2",
                    "question": "Changes in guidelines make me less confident in medical advice.",
                    "type": "likert_7"
                }
            ],
            "behavioral_intent": [
                {
                    "id": "b1",
                    "question": "I plan to follow the new recommendation.",
                    "type": "likert_7"
                },
                {
                    "id": "b2",
                    "question": "I want to get another doctor's opinion about this.",
                    "type": "likert_7"
                }
            ]
        }

        return questionnaire

    def generate_physician_checklist(self) -> List[Dict[str, Any]]:
        """
        Generate checklist for physician communication assessment.

        Returns
        -------
        List[Dict]
            Checklist items
        """
        change = self.guideline_change

        checklist = [
            {
                "id": "p1",
                "item": "Explained that the guideline has changed",
                "type": "yes_no"
            },
            {
                "id": "p2",
                "item": f"Mentioned the previous recommendation: {change.previous_recommendation}",
                "type": "yes_no"
            },
            {
                "id": "p3",
                "item": f"Clearly stated the new recommendation: {change.new_recommendation}",
                "type": "yes_no"
            },
            {
                "id": "p4",
                "item": "Explained why the guideline changed",
                "type": "yes_no"
            },
            {
                "id": "p5",
                "item": "Asked if patient had questions",
                "type": "yes_no"
            },
            {
                "id": "p6",
                "item": "Provided written materials about the change",
                "type": "yes_no"
            },
            {
                "id": "p7",
                "item": "Acknowledged that guideline changes can be confusing",
                "type": "yes_no"
            },
            {
                "id": "p8",
                "item": "Used teach-back to check understanding",
                "type": "yes_no"
            }
        ]

        return checklist

    def calculate_sample_size(
        self,
        expected_effect: float = 0.15,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> Dict[str, int]:
        """
        Calculate required sample size for study.

        Parameters
        ----------
        expected_effect : float
            Expected difference in confusion rate
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

        # Assuming baseline confusion rate of 0.3
        p1 = 0.30
        p2 = p1 - expected_effect

        # Sample size per group (intervention vs control)
        pooled_p = (p1 + p2) / 2
        n_per_group = (
            2 * ((z_alpha + z_beta) ** 2) * pooled_p * (1 - pooled_p)
        ) / (expected_effect ** 2)

        n_per_group = int(np.ceil(n_per_group))

        # Account for clustering (patients within physicians)
        icc = 0.05  # Intraclass correlation
        design_effect = 1 + (self.target_n_patients_per_physician - 1) * icc
        n_adjusted = int(np.ceil(n_per_group * design_effect))

        n_physicians = int(np.ceil(n_adjusted / self.target_n_patients_per_physician))

        return {
            "n_per_group_naive": n_per_group,
            "design_effect": design_effect,
            "n_per_group_adjusted": n_adjusted,
            "n_physicians_per_arm": n_physicians,
            "total_physicians": n_physicians * 2,
            "total_patients": n_adjusted * 2,
            "assumptions": {
                "baseline_confusion_rate": p1,
                "expected_reduction": expected_effect,
                "icc": icc,
                "patients_per_physician": self.target_n_patients_per_physician
            }
        }


# Pre-defined major medical reversals for study
NOTABLE_MEDICAL_REVERSALS = [
    TreatmentGuidelineChange(
        guideline_id="HRT_2002",
        condition="Menopause",
        treatment="Hormone Replacement Therapy",
        change_type=GuidelineChangeType.REVERSAL,
        change_date=datetime(2002, 7, 1),
        previous_recommendation="Recommended for all menopausal women",
        previous_evidence_level=EvidenceLevel.MODERATE,
        previous_target_population="All menopausal women",
        previous_duration_years=30,
        new_recommendation="Not recommended for disease prevention",
        new_evidence_level=EvidenceLevel.HIGH,
        new_target_population="Only for severe symptoms, short-term",
        triggering_evidence="WHI trial showed increased CVD and breast cancer risk",
        issuing_body="USPSTF",
        media_coverage_level="viral"
    ),
    TreatmentGuidelineChange(
        guideline_id="ASPIRIN_PRIMARY_2019",
        condition="Cardiovascular Disease Prevention",
        treatment="Daily Low-Dose Aspirin",
        change_type=GuidelineChangeType.REVERSAL,
        change_date=datetime(2019, 3, 1),
        previous_recommendation="Recommended for adults 50-70 at CVD risk",
        previous_evidence_level=EvidenceLevel.MODERATE,
        previous_target_population="Adults 50-70 with ≥10% 10-year CVD risk",
        previous_duration_years=20,
        new_recommendation="Not routinely recommended for primary prevention",
        new_evidence_level=EvidenceLevel.HIGH,
        new_target_population="Individual decision based on risk-benefit",
        triggering_evidence="ASPREE, ARRIVE, ASCEND trials showed bleeding risk outweighs benefit",
        issuing_body="USPSTF",
        media_coverage_level="high"
    ),
    TreatmentGuidelineChange(
        guideline_id="PSA_SCREENING_2012",
        condition="Prostate Cancer",
        treatment="PSA Screening",
        change_type=GuidelineChangeType.REVERSAL,
        change_date=datetime(2012, 5, 1),
        previous_recommendation="Annual PSA screening for men over 50",
        previous_evidence_level=EvidenceLevel.MODERATE,
        previous_target_population="Men over 50",
        previous_duration_years=15,
        new_recommendation="Do not routinely screen",
        new_evidence_level=EvidenceLevel.HIGH,
        new_target_population="Shared decision-making for 55-69",
        triggering_evidence="PLCO and ERSPC trials showed limited mortality benefit, significant harms",
        issuing_body="USPSTF",
        media_coverage_level="high"
    ),
    TreatmentGuidelineChange(
        guideline_id="DIETARY_FAT_2015",
        condition="Cardiovascular Health",
        treatment="Dietary Fat Restriction",
        change_type=GuidelineChangeType.REFINEMENT,
        change_date=datetime(2015, 1, 1),
        previous_recommendation="Limit total fat to <30% of calories",
        previous_evidence_level=EvidenceLevel.LOW,
        previous_target_population="All adults",
        previous_duration_years=35,
        new_recommendation="Focus on fat quality, not quantity",
        new_evidence_level=EvidenceLevel.MODERATE,
        new_target_population="All adults",
        triggering_evidence="Meta-analyses showing saturated fat focus misguided",
        issuing_body="Dietary Guidelines Advisory Committee",
        media_coverage_level="moderate"
    )
]
