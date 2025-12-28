"""
Longitudinal Study Analysis Module.

This module provides frameworks for analyzing the reversal curse
across extended time periods, tracking how expert knowledge
restructuring affects communication with novices over time.

Key capabilities:
1. Multi-session tracking of knowledge states
2. Temporal modeling of curse decay/persistence
3. Learning trajectory analysis
4. Expertise development modeling
5. Time-series analysis of asymmetry effects
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeSnapshot:
    """A snapshot of knowledge state at a point in time."""

    participant_id: str
    timestamp: datetime
    domain: str
    propositions: Dict[str, float]  # proposition -> belief strength
    test_performance: Dict[str, float]  # direction -> accuracy
    confidence_ratings: Dict[str, float]  # proposition -> confidence
    session_number: int
    time_since_reversal_days: Optional[float] = None

    @property
    def forward_accuracy(self) -> float:
        return self.test_performance.get("forward", 0.0)

    @property
    def reverse_accuracy(self) -> float:
        return self.test_performance.get("reverse", 0.0)

    @property
    def asymmetry_score(self) -> float:
        """Compute the asymmetry between forward and reverse performance."""
        return self.forward_accuracy - self.reverse_accuracy


@dataclass
class ReversalTimeline:
    """Timeline of a knowledge reversal event and its effects."""

    participant_id: str
    reversal_event_date: datetime
    domain: str
    reversal_description: str
    snapshots: List[KnowledgeSnapshot] = field(default_factory=list)

    @property
    def duration_days(self) -> float:
        if not self.snapshots:
            return 0.0
        last = max(s.timestamp for s in self.snapshots)
        return (last - self.reversal_event_date).total_seconds() / 86400

    @property
    def n_sessions(self) -> int:
        return len(self.snapshots)

    def get_asymmetry_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get time points and asymmetry scores."""
        times = []
        scores = []

        for snapshot in sorted(self.snapshots, key=lambda s: s.timestamp):
            days = (snapshot.timestamp - self.reversal_event_date).total_seconds() / 86400
            times.append(days)
            scores.append(snapshot.asymmetry_score)

        return np.array(times), np.array(scores)


@dataclass
class LongitudinalStudyConfig:
    """Configuration for longitudinal study."""

    study_name: str
    min_sessions_per_participant: int = 4
    max_session_gap_days: int = 30
    follow_up_intervals_days: List[int] = field(
        default_factory=lambda: [1, 7, 14, 30, 90]
    )
    require_baseline: bool = True
    domains: List[str] = field(default_factory=list)


class LongitudinalAnalyzer:
    """
    Analyzer for longitudinal reversal curse studies.

    Tracks how the curse effect evolves over time as:
    1. Initial shock fades
    2. New knowledge consolidates
    3. Theory of mind adapts
    4. Communication patterns develop
    """

    def __init__(self, config: Optional[LongitudinalStudyConfig] = None):
        """
        Initialize the analyzer.

        Parameters
        ----------
        config : LongitudinalStudyConfig, optional
            Study configuration
        """
        self.config = config or LongitudinalStudyConfig(study_name="default")
        self.timelines: Dict[str, ReversalTimeline] = {}
        self.fitted_models: Dict[str, Dict[str, Any]] = {}

    def add_timeline(self, timeline: ReversalTimeline) -> None:
        """Add a participant's reversal timeline."""
        key = f"{timeline.participant_id}_{timeline.domain}"
        self.timelines[key] = timeline

    def add_snapshot(
        self,
        participant_id: str,
        domain: str,
        snapshot: KnowledgeSnapshot
    ) -> None:
        """Add a knowledge snapshot to an existing timeline."""
        key = f"{participant_id}_{domain}"
        if key not in self.timelines:
            raise ValueError(f"No timeline found for {key}")
        self.timelines[key].snapshots.append(snapshot)

    def get_aggregate_trajectories(
        self,
        domain: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get aggregate asymmetry trajectories across participants.

        Parameters
        ----------
        domain : str, optional
            Filter by domain

        Returns
        -------
        pd.DataFrame
            Aggregated trajectory data
        """
        records = []

        for key, timeline in self.timelines.items():
            if domain and timeline.domain != domain:
                continue

            times, scores = timeline.get_asymmetry_trajectory()

            for t, s in zip(times, scores):
                records.append({
                    "participant_id": timeline.participant_id,
                    "domain": timeline.domain,
                    "days_since_reversal": t,
                    "asymmetry_score": s
                })

        return pd.DataFrame(records)

    def fit_decay_model(
        self,
        model_type: str = "exponential",
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fit a temporal decay model to the curse effect.

        Parameters
        ----------
        model_type : str
            Type of decay model: "exponential", "power", "hyperbolic"
        domain : str, optional
            Filter by domain

        Returns
        -------
        Dict[str, Any]
            Fitted model parameters and statistics
        """
        df = self.get_aggregate_trajectories(domain)

        if len(df) < 10:
            raise ValueError("Insufficient data for model fitting")

        x = df["days_since_reversal"].values
        y = df["asymmetry_score"].values

        # Define decay functions
        def exponential_decay(t, a, b, c):
            return a * np.exp(-b * t) + c

        def power_decay(t, a, b, c):
            return a * np.power(t + 1, -b) + c

        def hyperbolic_decay(t, a, b, c):
            return a / (1 + b * t) + c

        models = {
            "exponential": exponential_decay,
            "power": power_decay,
            "hyperbolic": hyperbolic_decay
        }

        model_func = models.get(model_type, exponential_decay)

        # Initial parameter guesses
        p0 = [y.max() - y.min(), 0.1, y.min()]

        try:
            popt, pcov = curve_fit(
                model_func, x, y, p0=p0,
                bounds=([0, 0, -1], [2, 10, 1]),
                maxfev=10000
            )

            # Compute fit statistics
            y_pred = model_func(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Parameter standard errors
            perr = np.sqrt(np.diag(pcov))

            # Compute half-life (time for curse to reduce by half)
            if model_type == "exponential":
                half_life = np.log(2) / popt[1] if popt[1] > 0 else np.inf
            elif model_type == "power":
                half_life = (2 ** (1 / popt[1]) - 1) if popt[1] > 0 else np.inf
            else:
                half_life = 1 / popt[1] if popt[1] > 0 else np.inf

            result = {
                "model_type": model_type,
                "parameters": {
                    "amplitude": popt[0],
                    "decay_rate": popt[1],
                    "asymptote": popt[2]
                },
                "parameter_errors": {
                    "amplitude_se": perr[0],
                    "decay_rate_se": perr[1],
                    "asymptote_se": perr[2]
                },
                "r_squared": r_squared,
                "half_life_days": half_life,
                "n_observations": len(df),
                "residual_std": np.sqrt(ss_res / len(df))
            }

            self.fitted_models[domain or "all"] = result
            return result

        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            return {"error": str(e)}

    def compare_decay_models(
        self,
        domain: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare different decay models.

        Parameters
        ----------
        domain : str, optional
            Filter by domain

        Returns
        -------
        Dict[str, Dict]
            Comparison of model fits
        """
        results = {}
        for model_type in ["exponential", "power", "hyperbolic"]:
            results[model_type] = self.fit_decay_model(model_type, domain)

        # Rank by R-squared
        valid_results = {
            k: v for k, v in results.items()
            if "error" not in v
        }

        if valid_results:
            best_model = max(
                valid_results.items(),
                key=lambda x: x[1]["r_squared"]
            )
            results["best_model"] = best_model[0]
            results["best_r_squared"] = best_model[1]["r_squared"]

        return results

    def analyze_individual_trajectories(
        self,
        domain: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze individual participant trajectories.

        Parameters
        ----------
        domain : str, optional
            Filter by domain

        Returns
        -------
        pd.DataFrame
            Individual trajectory statistics
        """
        records = []

        for key, timeline in self.timelines.items():
            if domain and timeline.domain != domain:
                continue

            times, scores = timeline.get_asymmetry_trajectory()

            if len(times) < 2:
                continue

            # Compute trajectory statistics
            initial_asymmetry = scores[0] if len(scores) > 0 else np.nan
            final_asymmetry = scores[-1] if len(scores) > 0 else np.nan
            max_asymmetry = np.max(scores)
            min_asymmetry = np.min(scores)

            # Linear trend
            if len(times) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(times, scores)
            else:
                slope = r_value = p_value = std_err = np.nan
                intercept = scores.mean()

            # Time to reach 50% reduction
            if initial_asymmetry > 0 and final_asymmetry < initial_asymmetry:
                target = initial_asymmetry / 2
                crossed_idx = np.where(scores <= target)[0]
                time_to_half = times[crossed_idx[0]] if len(crossed_idx) > 0 else np.nan
            else:
                time_to_half = np.nan

            records.append({
                "participant_id": timeline.participant_id,
                "domain": timeline.domain,
                "n_sessions": timeline.n_sessions,
                "duration_days": timeline.duration_days,
                "initial_asymmetry": initial_asymmetry,
                "final_asymmetry": final_asymmetry,
                "max_asymmetry": max_asymmetry,
                "min_asymmetry": min_asymmetry,
                "total_reduction": initial_asymmetry - final_asymmetry,
                "percent_reduction": (
                    (initial_asymmetry - final_asymmetry) / initial_asymmetry * 100
                    if initial_asymmetry > 0 else 0
                ),
                "trend_slope": slope,
                "trend_r_squared": r_value ** 2 if not np.isnan(r_value) else np.nan,
                "trend_p_value": p_value,
                "time_to_half_days": time_to_half
            })

        return pd.DataFrame(records)

    def identify_trajectory_clusters(
        self,
        n_clusters: int = 3,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Identify clusters of similar trajectory patterns.

        Parameters
        ----------
        n_clusters : int
            Number of clusters
        domain : str, optional
            Filter by domain

        Returns
        -------
        Dict[str, Any]
            Clustering results
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Get trajectory statistics
        df = self.analyze_individual_trajectories(domain)

        if len(df) < n_clusters:
            return {"error": "Insufficient data for clustering"}

        # Feature matrix for clustering
        features = df[[
            "initial_asymmetry", "final_asymmetry", "total_reduction",
            "trend_slope", "duration_days"
        ]].fillna(0).values

        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)

        df["cluster"] = clusters

        # Characterize clusters
        cluster_profiles = {}
        for c in range(n_clusters):
            cluster_df = df[df["cluster"] == c]
            cluster_profiles[f"cluster_{c}"] = {
                "n_participants": len(cluster_df),
                "mean_initial_asymmetry": cluster_df["initial_asymmetry"].mean(),
                "mean_final_asymmetry": cluster_df["final_asymmetry"].mean(),
                "mean_reduction": cluster_df["total_reduction"].mean(),
                "mean_duration": cluster_df["duration_days"].mean(),
                "profile_description": self._describe_cluster(cluster_df)
            }

        return {
            "n_clusters": n_clusters,
            "cluster_assignments": df[["participant_id", "domain", "cluster"]].to_dict("records"),
            "cluster_profiles": cluster_profiles
        }

    def _describe_cluster(self, cluster_df: pd.DataFrame) -> str:
        """Generate description for a trajectory cluster."""
        mean_initial = cluster_df["initial_asymmetry"].mean()
        mean_final = cluster_df["final_asymmetry"].mean()
        mean_reduction = cluster_df["total_reduction"].mean()

        if mean_reduction > 0.3:
            reduction_desc = "Strong recovery"
        elif mean_reduction > 0.1:
            reduction_desc = "Moderate recovery"
        else:
            reduction_desc = "Persistent curse"

        if mean_initial > 0.4:
            initial_desc = "high initial asymmetry"
        elif mean_initial > 0.2:
            initial_desc = "moderate initial asymmetry"
        else:
            initial_desc = "low initial asymmetry"

        return f"{reduction_desc} from {initial_desc}"


class ExpertiseDevelopmentTracker:
    """
    Track how expertise development affects the reversal curse.

    Models the relationship between:
    1. Accumulating domain expertise
    2. Vulnerability to the curse
    3. Recovery trajectories
    """

    def __init__(self):
        self.expertise_records: List[Dict[str, Any]] = []

    def record_expertise_level(
        self,
        participant_id: str,
        domain: str,
        timestamp: datetime,
        expertise_indicators: Dict[str, float],
        curse_susceptibility: float
    ) -> None:
        """
        Record expertise level and curse susceptibility.

        Parameters
        ----------
        participant_id : str
            Participant identifier
        domain : str
            Knowledge domain
        timestamp : datetime
            Time of measurement
        expertise_indicators : Dict[str, float]
            Various expertise metrics (years experience, test scores, etc.)
        curse_susceptibility : float
            Measured curse effect magnitude
        """
        self.expertise_records.append({
            "participant_id": participant_id,
            "domain": domain,
            "timestamp": timestamp,
            **expertise_indicators,
            "curse_susceptibility": curse_susceptibility
        })

    def get_expertise_curse_relationship(
        self,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze relationship between expertise and curse susceptibility.

        Parameters
        ----------
        domain : str, optional
            Filter by domain

        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        df = pd.DataFrame(self.expertise_records)

        if domain:
            df = df[df["domain"] == domain]

        if len(df) < 10:
            return {"error": "Insufficient data"}

        # Find expertise indicator columns
        indicator_cols = [
            c for c in df.columns
            if c not in ["participant_id", "domain", "timestamp", "curse_susceptibility"]
        ]

        results = {}
        for col in indicator_cols:
            if df[col].notna().sum() < 5:
                continue

            x = df[col].dropna().values
            y = df.loc[df[col].notna(), "curse_susceptibility"].values

            correlation, p_value = stats.pearsonr(x, y)

            # Also compute Spearman for non-linear relationships
            spearman_r, spearman_p = stats.spearmanr(x, y)

            results[col] = {
                "pearson_r": correlation,
                "pearson_p": p_value,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n_observations": len(x)
            }

        # Overall expertise composite
        if indicator_cols:
            composite = df[indicator_cols].mean(axis=1)
            overall_r, overall_p = stats.pearsonr(
                composite.values,
                df["curse_susceptibility"].values
            )
            results["composite_expertise"] = {
                "pearson_r": overall_r,
                "pearson_p": overall_p
            }

        return results

    def model_expertise_vulnerability_curve(
        self,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Model the expertise-vulnerability curve.

        Often follows an inverted-U: novices and true experts are less
        vulnerable than intermediate experts.

        Parameters
        ----------
        domain : str, optional
            Filter by domain

        Returns
        -------
        Dict[str, Any]
            Fitted curve parameters
        """
        df = pd.DataFrame(self.expertise_records)

        if domain:
            df = df[df["domain"] == domain]

        if len(df) < 20:
            return {"error": "Insufficient data for curve fitting"}

        # Find main expertise indicator
        indicator_cols = [
            c for c in df.columns
            if c not in ["participant_id", "domain", "timestamp", "curse_susceptibility"]
        ]

        if not indicator_cols:
            return {"error": "No expertise indicators found"}

        # Use first indicator or composite
        x = df[indicator_cols].mean(axis=1).values
        y = df["curse_susceptibility"].values

        # Fit inverted-U (quadratic)
        def inverted_u(x, a, b, c):
            return a * x ** 2 + b * x + c

        try:
            popt, pcov = curve_fit(inverted_u, x, y, p0=[-0.1, 0.5, 0.2])

            y_pred = inverted_u(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Find peak vulnerability
            peak_expertise = -popt[1] / (2 * popt[0]) if popt[0] != 0 else 0
            peak_vulnerability = inverted_u(peak_expertise, *popt)

            return {
                "model": "inverted_u",
                "parameters": {
                    "quadratic": popt[0],
                    "linear": popt[1],
                    "intercept": popt[2]
                },
                "r_squared": r_squared,
                "peak_expertise_level": peak_expertise,
                "peak_vulnerability": peak_vulnerability,
                "interpretation": (
                    f"Maximum vulnerability at expertise level {peak_expertise:.2f}, "
                    f"with vulnerability score {peak_vulnerability:.2f}"
                )
            }

        except Exception as e:
            return {"error": str(e)}


class RetentionAnalyzer:
    """
    Analyze retention of corrected knowledge over time.

    Tracks whether participants maintain the corrected understanding
    or revert to their original (incorrect) beliefs.
    """

    def __init__(self):
        self.retention_data: List[Dict[str, Any]] = []

    def record_retention_test(
        self,
        participant_id: str,
        domain: str,
        days_since_correction: float,
        correct_knowledge_score: float,
        reversion_indicators: Dict[str, float]
    ) -> None:
        """
        Record a retention test result.

        Parameters
        ----------
        participant_id : str
            Participant identifier
        domain : str
            Knowledge domain
        days_since_correction : float
            Days since knowledge correction
        correct_knowledge_score : float
            Score on correct knowledge test (0-1)
        reversion_indicators : Dict[str, float]
            Indicators of reversion to old beliefs
        """
        self.retention_data.append({
            "participant_id": participant_id,
            "domain": domain,
            "days_since_correction": days_since_correction,
            "correct_knowledge_score": correct_knowledge_score,
            **reversion_indicators
        })

    def fit_forgetting_curve(
        self,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fit Ebbinghaus-style forgetting curve.

        Parameters
        ----------
        domain : str, optional
            Filter by domain

        Returns
        -------
        Dict[str, Any]
            Forgetting curve parameters
        """
        df = pd.DataFrame(self.retention_data)

        if domain:
            df = df[df["domain"] == domain]

        if len(df) < 10:
            return {"error": "Insufficient data"}

        x = df["days_since_correction"].values
        y = df["correct_knowledge_score"].values

        # Ebbinghaus forgetting curve: R = e^(-t/S)
        def forgetting_curve(t, retention_strength, initial_retention):
            return initial_retention * np.exp(-t / retention_strength)

        try:
            popt, pcov = curve_fit(
                forgetting_curve, x, y,
                p0=[30, 0.9],
                bounds=([1, 0.5], [365, 1.0])
            )

            y_pred = forgetting_curve(x, *popt)
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

            return {
                "retention_strength_days": popt[0],
                "initial_retention": popt[1],
                "r_squared": r_squared,
                "retention_at_7_days": forgetting_curve(7, *popt),
                "retention_at_30_days": forgetting_curve(30, *popt),
                "retention_at_90_days": forgetting_curve(90, *popt),
                "interpretation": (
                    f"Knowledge decays with half-life of {popt[0] * np.log(2):.1f} days. "
                    f"Initial retention is {popt[1] * 100:.1f}%."
                )
            }

        except Exception as e:
            return {"error": str(e)}

    def identify_at_risk_participants(
        self,
        threshold: float = 0.5,
        domain: Optional[str] = None
    ) -> List[str]:
        """
        Identify participants at risk of knowledge reversion.

        Parameters
        ----------
        threshold : float
            Score below which is considered at risk
        domain : str, optional
            Filter by domain

        Returns
        -------
        List[str]
            Participant IDs at risk
        """
        df = pd.DataFrame(self.retention_data)

        if domain:
            df = df[df["domain"] == domain]

        # Get most recent test for each participant
        recent = df.sort_values("days_since_correction").groupby("participant_id").last()

        at_risk = recent[recent["correct_knowledge_score"] < threshold].index.tolist()

        return at_risk


def create_longitudinal_study(
    study_name: str,
    domains: List[str],
    follow_up_schedule: Optional[List[int]] = None
) -> Tuple[LongitudinalStudyConfig, LongitudinalAnalyzer]:
    """
    Convenience function to create a longitudinal study setup.

    Parameters
    ----------
    study_name : str
        Name of the study
    domains : List[str]
        Knowledge domains to study
    follow_up_schedule : List[int], optional
        Follow-up intervals in days

    Returns
    -------
    Tuple[LongitudinalStudyConfig, LongitudinalAnalyzer]
        Configuration and analyzer
    """
    config = LongitudinalStudyConfig(
        study_name=study_name,
        domains=domains,
        follow_up_intervals_days=follow_up_schedule or [1, 7, 14, 30, 90]
    )

    analyzer = LongitudinalAnalyzer(config)

    return config, analyzer
