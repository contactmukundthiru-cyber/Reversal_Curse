"""
Neural Correlates Analysis Framework for Reversal Curse Research.

This module provides frameworks for analyzing neuroimaging data
(fMRI, EEG) related to knowledge restructuring during reversals.

Key research questions:
1. What brain regions show differential activation during reversals?
2. Are there neural signatures that predict curse susceptibility?
3. Do successful debiasing interventions show distinct neural patterns?

The framework supports:
- fMRI region-of-interest analysis
- EEG event-related potential analysis
- Connectivity analysis during knowledge restructuring
- Neural prediction of behavioral outcomes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class ImagingModality(Enum):
    """Neuroimaging modalities supported."""

    FMRI = "fmri"
    EEG = "eeg"
    FNIRS = "fnirs"
    MEG = "meg"


class BrainRegion(Enum):
    """Key brain regions for reversal curse analysis."""

    # Prefrontal regions (executive control, belief updating)
    DLPFC_LEFT = "dlpfc_left"
    DLPFC_RIGHT = "dlpfc_right"
    VMPFC = "vmpfc"
    ACC = "acc"  # Anterior cingulate cortex (conflict monitoring)
    OFC = "ofc"  # Orbitofrontal cortex

    # Parietal regions (belief integration)
    TPJ_LEFT = "tpj_left"  # Theory of mind
    TPJ_RIGHT = "tpj_right"
    PPC = "ppc"  # Posterior parietal cortex
    PRECUNEUS = "precuneus"

    # Temporal regions (semantic memory)
    ATL_LEFT = "atl_left"  # Anterior temporal lobe
    ATL_RIGHT = "atl_right"
    HIPPOCAMPUS_LEFT = "hippocampus_left"
    HIPPOCAMPUS_RIGHT = "hippocampus_right"

    # Other
    INSULA_LEFT = "insula_left"
    INSULA_RIGHT = "insula_right"
    AMYGDALA_LEFT = "amygdala_left"
    AMYGDALA_RIGHT = "amygdala_right"


@dataclass
class MNICoordinates:
    """MNI coordinates for brain regions."""

    region: BrainRegion
    x: float
    y: float
    z: float
    radius_mm: float = 8.0  # Sphere radius for ROI


# Standard MNI coordinates for key regions
STANDARD_ROI_COORDINATES = {
    BrainRegion.DLPFC_LEFT: MNICoordinates(BrainRegion.DLPFC_LEFT, -44, 36, 20),
    BrainRegion.DLPFC_RIGHT: MNICoordinates(BrainRegion.DLPFC_RIGHT, 44, 36, 20),
    BrainRegion.VMPFC: MNICoordinates(BrainRegion.VMPFC, 0, 44, -8),
    BrainRegion.ACC: MNICoordinates(BrainRegion.ACC, 0, 30, 28),
    BrainRegion.TPJ_LEFT: MNICoordinates(BrainRegion.TPJ_LEFT, -52, -56, 24),
    BrainRegion.TPJ_RIGHT: MNICoordinates(BrainRegion.TPJ_RIGHT, 52, -56, 24),
    BrainRegion.HIPPOCAMPUS_LEFT: MNICoordinates(BrainRegion.HIPPOCAMPUS_LEFT, -26, -20, -16),
    BrainRegion.HIPPOCAMPUS_RIGHT: MNICoordinates(BrainRegion.HIPPOCAMPUS_RIGHT, 26, -20, -16),
    BrainRegion.PRECUNEUS: MNICoordinates(BrainRegion.PRECUNEUS, 0, -60, 40),
}


@dataclass
class NeuralEvent:
    """A neural event during the experiment."""

    participant_id: str
    event_type: str  # "pre_reversal", "reversal_presentation", "post_reversal_test"
    timestamp_ms: float
    trial_number: int
    condition: str  # "forward", "reverse", "simultaneous"
    behavioral_response: Optional[str] = None
    reaction_time_ms: Optional[float] = None
    correct: Optional[bool] = None


@dataclass
class fMRISession:
    """Data from an fMRI session."""

    participant_id: str
    session_date: datetime
    tr_ms: float  # Repetition time
    n_volumes: int
    n_runs: int

    # ROI time series (region -> array of shape (n_volumes, n_runs))
    roi_timeseries: Dict[BrainRegion, np.ndarray] = field(default_factory=dict)

    # Events
    events: List[NeuralEvent] = field(default_factory=list)

    # Motion parameters (volumes x 6)
    motion_params: Optional[np.ndarray] = None

    # Quality metrics
    mean_framewise_displacement: Optional[float] = None
    n_censored_volumes: Optional[int] = None


@dataclass
class EEGSession:
    """Data from an EEG session."""

    participant_id: str
    session_date: datetime
    sampling_rate_hz: float
    n_channels: int
    channel_names: List[str]

    # Raw EEG data (channels x time) - typically preprocessed
    data: Optional[np.ndarray] = None

    # Events
    events: List[NeuralEvent] = field(default_factory=list)

    # Epochs (n_epochs x n_channels x n_timepoints)
    epochs: Optional[np.ndarray] = None
    epoch_times_ms: Optional[np.ndarray] = None
    epoch_conditions: Optional[List[str]] = None

    # Quality metrics
    n_bad_channels: int = 0
    n_rejected_epochs: int = 0
    ica_components_removed: int = 0


class fMRIAnalyzer:
    """
    Analyzer for fMRI data related to knowledge restructuring.

    Focuses on:
    1. ROI activation during reversals vs. consistent trials
    2. Connectivity changes during knowledge restructuring
    3. Neural predictors of reversal curse susceptibility
    """

    def __init__(self):
        self.sessions: Dict[str, fMRISession] = {}
        self.group_results: Dict[str, Any] = {}

    def add_session(self, session: fMRISession) -> None:
        """Add an fMRI session."""
        self.sessions[session.participant_id] = session

    def extract_trial_betas(
        self,
        session: fMRISession,
        event_type: str,
        pre_onset_volumes: int = 0,
        post_onset_volumes: int = 8
    ) -> Dict[BrainRegion, np.ndarray]:
        """
        Extract trial-wise activation estimates (betas).

        Parameters
        ----------
        session : fMRISession
            The session to analyze
        event_type : str
            Type of event to extract
        pre_onset_volumes : int
            Volumes before event onset
        post_onset_volumes : int
            Volumes after event onset (including HRF lag)

        Returns
        -------
        Dict[BrainRegion, np.ndarray]
            Beta estimates per region (n_trials,)
        """
        relevant_events = [e for e in session.events if e.event_type == event_type]

        betas = {}
        tr_s = session.tr_ms / 1000

        for region, timeseries in session.roi_timeseries.items():
            trial_betas = []

            for event in relevant_events:
                # Convert event time to volume index
                onset_volume = int(event.timestamp_ms / 1000 / tr_s)

                # Extract window
                start = max(0, onset_volume - pre_onset_volumes)
                end = min(timeseries.shape[0], onset_volume + post_onset_volumes)

                if end > start:
                    # Simple beta: mean activation in window minus baseline
                    window_mean = timeseries[start:end].mean()
                    baseline = timeseries[max(0, onset_volume - 4):onset_volume].mean() if onset_volume > 0 else 0
                    trial_betas.append(window_mean - baseline)

            betas[region] = np.array(trial_betas)

        return betas

    def compare_conditions(
        self,
        region: BrainRegion,
        condition_a: str = "reversal_presentation",
        condition_b: str = "consistent_presentation"
    ) -> Dict[str, Any]:
        """
        Compare activation between conditions across participants.

        Parameters
        ----------
        region : BrainRegion
            Brain region to analyze
        condition_a : str
            First condition
        condition_b : str
            Second condition

        Returns
        -------
        Dict[str, Any]
            Statistical comparison results
        """
        activation_a = []
        activation_b = []

        for pid, session in self.sessions.items():
            betas = self.extract_trial_betas(session, condition_a)
            if region in betas and len(betas[region]) > 0:
                activation_a.append(betas[region].mean())

            betas = self.extract_trial_betas(session, condition_b)
            if region in betas and len(betas[region]) > 0:
                activation_b.append(betas[region].mean())

        if len(activation_a) < 5 or len(activation_b) < 5:
            return {"error": "Insufficient data"}

        activation_a = np.array(activation_a)
        activation_b = np.array(activation_b)

        # Paired t-test (within-subject design)
        t_stat, p_value = stats.ttest_rel(activation_a, activation_b)

        # Effect size (Cohen's d for paired samples)
        diff = activation_a - activation_b
        cohens_d = diff.mean() / diff.std()

        return {
            "region": region.value,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "mean_a": float(activation_a.mean()),
            "mean_b": float(activation_b.mean()),
            "difference": float(diff.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "n_participants": len(activation_a),
            "significant": p_value < 0.05
        }

    def analyze_hypothesis_regions(self) -> Dict[str, Any]:
        """
        Test hypotheses about specific brain regions.

        Key hypotheses:
        1. ACC shows increased activation during reversal (conflict monitoring)
        2. VMPFC shows decreased activation (belief updating difficulty)
        3. TPJ shows differential activation related to ToM curse
        4. Hippocampus shows restructuring signatures

        Returns
        -------
        Dict[str, Any]
            Results for each hypothesis
        """
        results = {}

        # H1: ACC conflict signal during reversals
        results["H1_ACC_conflict"] = self.compare_conditions(
            BrainRegion.ACC,
            "reversal_presentation",
            "consistent_presentation"
        )

        # H2: VMPFC belief updating
        results["H2_VMPFC_updating"] = self.compare_conditions(
            BrainRegion.VMPFC,
            "reversal_presentation",
            "consistent_presentation"
        )

        # H3: TPJ theory of mind
        results["H3_TPJ_tom"] = self.compare_conditions(
            BrainRegion.TPJ_RIGHT,
            "reversal_presentation",
            "consistent_presentation"
        )

        # H4: Hippocampal restructuring
        results["H4_hippocampus"] = self.compare_conditions(
            BrainRegion.HIPPOCAMPUS_LEFT,
            "reversal_presentation",
            "consistent_presentation"
        )

        return results

    def predict_behavioral_from_neural(
        self,
        behavioral_metric: str = "curse_magnitude"
    ) -> Dict[str, Any]:
        """
        Use neural activation to predict behavioral outcomes.

        Parameters
        ----------
        behavioral_metric : str
            Behavioral metric to predict

        Returns
        -------
        Dict[str, Any]
            Prediction results
        """
        # Collect neural and behavioral data
        neural_features = []
        behavioral_scores = []

        for pid, session in self.sessions.items():
            # Get mean activation in key regions during reversal
            reversal_betas = self.extract_trial_betas(session, "reversal_presentation")

            if not reversal_betas:
                continue

            # Feature vector: activation in hypothesis regions
            features = []
            for region in [BrainRegion.ACC, BrainRegion.VMPFC, BrainRegion.TPJ_RIGHT,
                          BrainRegion.HIPPOCAMPUS_LEFT, BrainRegion.DLPFC_LEFT]:
                if region in reversal_betas and len(reversal_betas[region]) > 0:
                    features.append(reversal_betas[region].mean())
                else:
                    features.append(0)

            # Get behavioral score
            reversal_trials = [e for e in session.events if e.event_type == "post_reversal_test"]
            if reversal_trials:
                accuracy = np.mean([t.correct for t in reversal_trials if t.correct is not None])
                # Curse magnitude = 1 - accuracy on reversal trials
                behavioral_scores.append(1 - accuracy)
                neural_features.append(features)

        if len(behavioral_scores) < 10:
            return {"error": "Insufficient data for prediction"}

        X = np.array(neural_features)
        y = np.array(behavioral_scores)

        # Multiple regression
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        model = LinearRegression()
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        # Fit final model for coefficients
        model.fit(X, y)

        region_names = ["ACC", "VMPFC", "TPJ_R", "Hipp_L", "DLPFC_L"]
        coefficients = dict(zip(region_names, model.coef_))

        return {
            "cv_r2_scores": cv_scores.tolist(),
            "mean_cv_r2": float(cv_scores.mean()),
            "std_cv_r2": float(cv_scores.std()),
            "coefficients": coefficients,
            "intercept": float(model.intercept_),
            "n_participants": len(y),
            "feature_names": region_names
        }


class EEGAnalyzer:
    """
    Analyzer for EEG data related to knowledge restructuring.

    Focuses on:
    1. ERPs during reversal processing
    2. Oscillatory signatures of belief updating
    3. Neural markers of successful vs. failed updating
    """

    def __init__(self):
        self.sessions: Dict[str, EEGSession] = {}
        self.group_results: Dict[str, Any] = {}

    def add_session(self, session: EEGSession) -> None:
        """Add an EEG session."""
        self.sessions[session.participant_id] = session

    def compute_erp(
        self,
        session: EEGSession,
        condition: str,
        channels: Optional[List[str]] = None,
        baseline_ms: Tuple[float, float] = (-200, 0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute event-related potential for a condition.

        Parameters
        ----------
        session : EEGSession
            The session to analyze
        condition : str
            Condition to extract
        channels : List[str], optional
            Channels to include (default: all)
        baseline_ms : Tuple[float, float]
            Baseline window for correction

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ERP (channels x time) and time points
        """
        if session.epochs is None or session.epoch_conditions is None:
            raise ValueError("Session must have epoched data")

        # Select epochs for condition
        condition_mask = np.array([c == condition for c in session.epoch_conditions])
        condition_epochs = session.epochs[condition_mask]

        if len(condition_epochs) == 0:
            raise ValueError(f"No epochs found for condition {condition}")

        times = session.epoch_times_ms

        # Baseline correction
        baseline_mask = (times >= baseline_ms[0]) & (times < baseline_ms[1])
        if baseline_mask.any():
            baseline_mean = condition_epochs[:, :, baseline_mask].mean(axis=2, keepdims=True)
            condition_epochs = condition_epochs - baseline_mean

        # Average across trials
        erp = condition_epochs.mean(axis=0)

        # Channel selection
        if channels:
            channel_idx = [session.channel_names.index(c) for c in channels if c in session.channel_names]
            erp = erp[channel_idx]

        return erp, times

    def analyze_p300(
        self,
        condition_reversal: str = "reversal_presentation",
        condition_consistent: str = "consistent_presentation",
        window_ms: Tuple[float, float] = (300, 500),
        channels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze P300 component, which reflects context updating.

        The P300 is expected to be larger for reversals (more updating needed).

        Parameters
        ----------
        condition_reversal : str
            Reversal condition name
        condition_consistent : str
            Consistent condition name
        window_ms : Tuple[float, float]
            Time window for P300 measurement
        channels : List[str], optional
            Parietal channels to analyze

        Returns
        -------
        Dict[str, Any]
            P300 analysis results
        """
        if channels is None:
            channels = ["Pz", "P3", "P4", "CPz"]

        reversal_p300s = []
        consistent_p300s = []

        for pid, session in self.sessions.items():
            try:
                # Get ERPs
                erp_rev, times = self.compute_erp(session, condition_reversal, channels)
                erp_con, _ = self.compute_erp(session, condition_consistent, channels)

                # Extract P300 amplitude
                window_mask = (times >= window_ms[0]) & (times <= window_ms[1])
                p300_rev = erp_rev[:, window_mask].mean()
                p300_con = erp_con[:, window_mask].mean()

                reversal_p300s.append(p300_rev)
                consistent_p300s.append(p300_con)

            except (ValueError, KeyError) as e:
                logger.warning(f"Could not compute P300 for {pid}: {e}")
                continue

        if len(reversal_p300s) < 5:
            return {"error": "Insufficient data"}

        reversal_p300s = np.array(reversal_p300s)
        consistent_p300s = np.array(consistent_p300s)

        # Statistical comparison
        t_stat, p_value = stats.ttest_rel(reversal_p300s, consistent_p300s)
        diff = reversal_p300s - consistent_p300s
        cohens_d = diff.mean() / diff.std()

        return {
            "component": "P300",
            "window_ms": window_ms,
            "channels": channels,
            "mean_reversal_uV": float(reversal_p300s.mean()),
            "mean_consistent_uV": float(consistent_p300s.mean()),
            "difference_uV": float(diff.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "n_participants": len(reversal_p300s),
            "interpretation": (
                "Larger P300 for reversals suggests greater context updating demand"
                if diff.mean() > 0 else
                "No P300 enhancement for reversals"
            )
        }

    def analyze_n400(
        self,
        condition_reversal: str = "reversal_presentation",
        condition_consistent: str = "consistent_presentation",
        window_ms: Tuple[float, float] = (350, 500),
        channels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze N400 component, which reflects semantic processing.

        The N400 should be larger for semantic violations during reversal.

        Parameters
        ----------
        condition_reversal : str
            Reversal condition name
        condition_consistent : str
            Consistent condition name
        window_ms : Tuple[float, float]
            Time window for N400 measurement
        channels : List[str], optional
            Central-parietal channels

        Returns
        -------
        Dict[str, Any]
            N400 analysis results
        """
        if channels is None:
            channels = ["Cz", "CPz", "Pz"]

        reversal_n400s = []
        consistent_n400s = []

        for pid, session in self.sessions.items():
            try:
                erp_rev, times = self.compute_erp(session, condition_reversal, channels)
                erp_con, _ = self.compute_erp(session, condition_consistent, channels)

                window_mask = (times >= window_ms[0]) & (times <= window_ms[1])
                n400_rev = erp_rev[:, window_mask].mean()  # N400 is negative
                n400_con = erp_con[:, window_mask].mean()

                reversal_n400s.append(n400_rev)
                consistent_n400s.append(n400_con)

            except (ValueError, KeyError) as e:
                logger.warning(f"Could not compute N400 for {pid}: {e}")
                continue

        if len(reversal_n400s) < 5:
            return {"error": "Insufficient data"}

        reversal_n400s = np.array(reversal_n400s)
        consistent_n400s = np.array(consistent_n400s)

        t_stat, p_value = stats.ttest_rel(reversal_n400s, consistent_n400s)
        diff = reversal_n400s - consistent_n400s  # More negative = larger N400
        cohens_d = diff.mean() / diff.std()

        return {
            "component": "N400",
            "window_ms": window_ms,
            "channels": channels,
            "mean_reversal_uV": float(reversal_n400s.mean()),
            "mean_consistent_uV": float(consistent_n400s.mean()),
            "difference_uV": float(diff.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "n_participants": len(reversal_n400s),
            "interpretation": (
                "Enhanced N400 for reversals indicates semantic conflict processing"
                if diff.mean() < 0 else
                "No N400 enhancement for reversals"
            )
        }

    def analyze_theta_power(
        self,
        condition: str,
        frequency_band: Tuple[float, float] = (4, 8),
        channels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze frontal theta power, associated with cognitive control.

        Parameters
        ----------
        condition : str
            Condition to analyze
        frequency_band : Tuple[float, float]
            Theta frequency range (Hz)
        channels : List[str], optional
            Frontal channels

        Returns
        -------
        Dict[str, Any]
            Theta power analysis
        """
        if channels is None:
            channels = ["Fz", "FCz", "Cz"]

        theta_powers = []

        for pid, session in self.sessions.items():
            if session.epochs is None:
                continue

            # Get epochs for condition
            condition_mask = np.array([c == condition for c in session.epoch_conditions])
            condition_epochs = session.epochs[condition_mask]

            if len(condition_epochs) == 0:
                continue

            # Channel selection
            channel_idx = [
                session.channel_names.index(c)
                for c in channels
                if c in session.channel_names
            ]

            if not channel_idx:
                continue

            channel_data = condition_epochs[:, channel_idx, :]

            # Compute power spectral density
            fs = session.sampling_rate_hz
            freqs, psd = signal.welch(channel_data, fs, nperseg=256, axis=-1)

            # Extract theta band
            theta_mask = (freqs >= frequency_band[0]) & (freqs <= frequency_band[1])
            theta_power = psd[:, :, theta_mask].mean()

            theta_powers.append(theta_power)

        if len(theta_powers) < 5:
            return {"error": "Insufficient data"}

        theta_powers = np.array(theta_powers)

        return {
            "condition": condition,
            "frequency_band_hz": frequency_band,
            "channels": channels,
            "mean_theta_power": float(theta_powers.mean()),
            "std_theta_power": float(theta_powers.std()),
            "n_participants": len(theta_powers)
        }


@dataclass
class NeuralStudyDesign:
    """
    Design for a neuroimaging study of the reversal curse.

    Provides structured methodology for fMRI or EEG experiments.
    """

    study_name: str
    modality: ImagingModality
    target_n_participants: int = 30
    n_runs: int = 4  # fMRI runs or EEG blocks
    trials_per_condition: int = 40

    # Timing (in seconds for fMRI, milliseconds for EEG)
    stimulus_duration: float = 2.0
    isi_range: Tuple[float, float] = (2.0, 6.0)  # Jittered ISI

    # Conditions
    conditions: List[str] = field(default_factory=lambda: [
        "reversal_forward",  # Forward direction after reversal
        "reversal_reverse",  # Reverse direction after reversal
        "consistent_forward",  # Forward direction (no reversal)
        "consistent_reverse",  # Reverse direction (no reversal)
    ])

    # ROIs for fMRI
    hypothesis_rois: List[BrainRegion] = field(default_factory=lambda: [
        BrainRegion.ACC,
        BrainRegion.VMPFC,
        BrainRegion.TPJ_RIGHT,
        BrainRegion.HIPPOCAMPUS_LEFT,
        BrainRegion.DLPFC_LEFT,
    ])

    def generate_stimulus_timing(self, run_number: int) -> List[Dict[str, Any]]:
        """
        Generate stimulus timing for a run.

        Parameters
        ----------
        run_number : int
            Run number (for randomization seed)

        Returns
        -------
        List[Dict]
            Stimulus timing information
        """
        np.random.seed(42 + run_number)

        n_trials_per_run = self.trials_per_condition * len(self.conditions) // self.n_runs
        trials_per_condition_per_run = n_trials_per_run // len(self.conditions)

        events = []
        current_time = 10.0  # Initial fixation

        # Create trial list
        trial_conditions = []
        for cond in self.conditions:
            trial_conditions.extend([cond] * trials_per_condition_per_run)

        np.random.shuffle(trial_conditions)

        for i, condition in enumerate(trial_conditions):
            # Jittered ISI
            isi = np.random.uniform(*self.isi_range)

            events.append({
                "trial_number": i + 1,
                "condition": condition,
                "onset": current_time,
                "duration": self.stimulus_duration,
            })

            current_time += self.stimulus_duration + isi

        return events

    def calculate_power(
        self,
        expected_effect_size: float = 0.5,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate statistical power for the study.

        Parameters
        ----------
        expected_effect_size : float
            Expected Cohen's d
        alpha : float
            Significance level

        Returns
        -------
        Dict[str, Any]
            Power analysis results
        """
        from scipy.stats import nct

        # Degrees of freedom for paired t-test
        df = self.target_n_participants - 1

        # Non-centrality parameter
        ncp = expected_effect_size * np.sqrt(self.target_n_participants)

        # Critical t-value
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        # Power
        power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)

        # Required N for 80% power
        def power_at_n(n):
            df = n - 1
            ncp = expected_effect_size * np.sqrt(n)
            t_crit = stats.t.ppf(1 - alpha / 2, df)
            return 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)

        required_n = self.target_n_participants
        while power_at_n(required_n) < 0.8 and required_n < 200:
            required_n += 1

        return {
            "target_n": self.target_n_participants,
            "expected_effect_size": expected_effect_size,
            "alpha": alpha,
            "achieved_power": float(power),
            "required_n_for_80_power": required_n,
            "total_trials_per_participant": self.trials_per_condition * len(self.conditions),
            "trials_per_condition": self.trials_per_condition
        }

    def generate_hypotheses(self) -> List[Dict[str, str]]:
        """
        Generate formal hypotheses for the study.

        Returns
        -------
        List[Dict[str, str]]
            Hypotheses with descriptions
        """
        return [
            {
                "id": "H1",
                "name": "ACC Conflict Monitoring",
                "hypothesis": "ACC activation will be greater for reversal vs. consistent trials",
                "rationale": "ACC is involved in conflict monitoring and error detection; "
                            "reversals create cognitive conflict",
                "analysis": "Paired t-test on ACC ROI betas: reversal > consistent"
            },
            {
                "id": "H2",
                "name": "VMPFC Belief Updating",
                "hypothesis": "VMPFC will show differential activation during reversal processing",
                "rationale": "VMPFC is involved in value updating and belief revision",
                "analysis": "Paired t-test on VMPFC ROI betas: reversal vs. consistent"
            },
            {
                "id": "H3",
                "name": "TPJ Theory of Mind",
                "hypothesis": "TPJ activation during reversal will predict subsequent ToM performance",
                "rationale": "TPJ supports perspective-taking; its engagement during reversal may "
                            "buffer against the curse",
                "analysis": "Correlation: TPJ activation ~ ToM task accuracy"
            },
            {
                "id": "H4",
                "name": "Hippocampal Restructuring",
                "hypothesis": "Hippocampal activation will correlate with successful knowledge updating",
                "rationale": "Hippocampus is critical for memory reorganization",
                "analysis": "Correlation: Hippocampus activation ~ post-reversal accuracy improvement"
            },
            {
                "id": "H5",
                "name": "Frontal Theta for Cognitive Control",
                "hypothesis": "Frontal theta power will be enhanced during reversal processing",
                "rationale": "Theta oscillations support cognitive control processes",
                "analysis": "Paired t-test on theta power (4-8 Hz): reversal > consistent"
            }
        ]


def create_neural_study(
    study_name: str,
    modality: str = "fmri",
    n_participants: int = 30
) -> Tuple[NeuralStudyDesign, Union[fMRIAnalyzer, EEGAnalyzer]]:
    """
    Convenience function to create a neural study setup.

    Parameters
    ----------
    study_name : str
        Name of the study
    modality : str
        "fmri" or "eeg"
    n_participants : int
        Target sample size

    Returns
    -------
    Tuple[NeuralStudyDesign, Analyzer]
        Study design and appropriate analyzer
    """
    modality_enum = ImagingModality.FMRI if modality.lower() == "fmri" else ImagingModality.EEG

    design = NeuralStudyDesign(
        study_name=study_name,
        modality=modality_enum,
        target_n_participants=n_participants
    )

    analyzer = fMRIAnalyzer() if modality.lower() == "fmri" else EEGAnalyzer()

    return design, analyzer
