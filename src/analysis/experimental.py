"""
Experimental data analysis pipeline for Study 3.

This module provides:
- Analysis of the controlled experiment
- 2x3 mixed ANOVA implementation
- Planned contrasts and follow-up tests
- Power analysis and effect size calculations
- Publication-ready results formatting

Study Design:
- 2 (Test direction: Forward vs. Reverse) × 3 (Training: A-then-B vs. B-then-A vs. Simultaneous)
- Test direction is within-subjects
- Training condition is between-subjects
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import get_config
from .statistics import (
    calculate_reversal_gap,
    compute_cohens_h,
    compute_cohens_d,
    bootstrap_ci,
    equivalence_test,
    anova_2x3_mixed,
    mcnemar_test,
    two_proportion_test,
    StatisticalResult,
    ReversalGapResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ParticipantData:
    """Data for a single participant."""

    participant_id: str
    condition: str  # A_then_B, B_then_A, simultaneous
    forward_correct: int
    forward_total: int
    reverse_correct: int
    reverse_total: int
    trials_to_criterion: int
    completion_time_seconds: float
    passed_attention_check: bool
    demographics: Dict[str, Any]


@dataclass
class ConditionResults:
    """Results for a single experimental condition."""

    condition: str
    n_participants: int
    forward_accuracy: float
    forward_ci: Tuple[float, float]
    reverse_accuracy: float
    reverse_ci: Tuple[float, float]
    asymmetry: float
    asymmetry_ci: Tuple[float, float]
    within_test: StatisticalResult


@dataclass
class ExperimentalResults:
    """Complete experimental results."""

    n_total: int
    n_per_condition: Dict[str, int]
    n_excluded: int
    exclusion_reasons: Dict[str, int]
    condition_results: Dict[str, ConditionResults]
    anova_results: Dict[str, StatisticalResult]
    flip_test: StatisticalResult
    equivalence_test: Dict[str, Any]
    effect_sizes: Dict[str, float]


class ExperimentalAnalyzer:
    """
    Analyzer for the controlled experiment (Study 3).

    This class implements the pre-registered analysis plan:
    1. Primary analysis: 2×3 mixed ANOVA
    2. Planned contrasts within each condition
    3. Flip test: comparing asymmetry between A-then-B and B-then-A
    4. Equivalence test for simultaneous condition
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the analyzer.

        Parameters
        ----------
        config : Optional[Any]
            Configuration object. Uses default if None.
        """
        self.config = config or get_config().experiment
        self.raw_data: Optional[pd.DataFrame] = None
        self.clean_data: Optional[pd.DataFrame] = None
        self.results: Optional[ExperimentalResults] = None

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """
        Load experimental data.

        Expected columns:
        - participant_id: Unique identifier
        - condition: A_then_B, B_then_A, simultaneous
        - forward_correct: Number of correct forward trials
        - forward_total: Total forward trials
        - reverse_correct: Number of correct reverse trials
        - reverse_total: Total reverse trials
        - trials_to_criterion: Training trials to reach criterion
        - completion_time: Total time in seconds
        - attention_check_passed: Boolean
        - age, gender, etc.

        Parameters
        ----------
        data_path : Path
            Path to data file

        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        logger.info(f"Loading experimental data from {data_path}")

        if data_path.suffix == ".parquet":
            self.raw_data = pd.read_parquet(data_path)
        elif data_path.suffix == ".json":
            self.raw_data = pd.read_json(data_path)
        else:
            self.raw_data = pd.read_csv(data_path)

        logger.info(f"Loaded {len(self.raw_data)} participants")
        return self.raw_data

    def apply_exclusions(self) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Apply pre-registered exclusion criteria.

        Exclusion criteria:
        1. Failed attention check
        2. <50% accuracy on trained direction
        3. Completion time <5 minutes

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, int]]
            Clean data and exclusion counts
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.raw_data.copy()
        exclusion_reasons = {
            "attention_check": 0,
            "low_trained_accuracy": 0,
            "too_fast": 0,
            "total_excluded": 0,
        }

        initial_n = len(df)

        # Calculate trained direction accuracy
        df["trained_accuracy"] = df.apply(
            lambda row: (
                row["forward_correct"] / row["forward_total"]
                if row["condition"] == "A_then_B"
                else row["reverse_correct"] / row["reverse_total"]
                if row["condition"] == "B_then_A"
                else (row["forward_correct"] + row["reverse_correct"]) /
                     (row["forward_total"] + row["reverse_total"])
            ),
            axis=1
        )

        # Apply exclusions
        attention_mask = df["attention_check_passed"] == True
        exclusion_reasons["attention_check"] = (~attention_mask).sum()

        accuracy_mask = df["trained_accuracy"] >= self.config.min_trained_direction_accuracy
        exclusion_reasons["low_trained_accuracy"] = (
            attention_mask & ~accuracy_mask
        ).sum()

        min_time = self.config.min_completion_time_minutes * 60
        time_mask = df["completion_time"] >= min_time
        exclusion_reasons["too_fast"] = (
            attention_mask & accuracy_mask & ~time_mask
        ).sum()

        # Combined mask
        keep_mask = attention_mask & accuracy_mask & time_mask
        self.clean_data = df[keep_mask].copy()

        exclusion_reasons["total_excluded"] = initial_n - len(self.clean_data)

        logger.info(
            f"After exclusions: {len(self.clean_data)} participants "
            f"({exclusion_reasons['total_excluded']} excluded)"
        )

        return self.clean_data, exclusion_reasons

    def compute_condition_results(self) -> Dict[str, ConditionResults]:
        """
        Compute descriptive statistics and within-condition tests.

        Returns
        -------
        Dict[str, ConditionResults]
            Results for each condition
        """
        if self.clean_data is None:
            self.apply_exclusions()

        df = self.clean_data
        results = {}

        for condition in self.config.conditions:
            cond_df = df[df["condition"] == condition]

            if len(cond_df) == 0:
                continue

            # Accuracies
            forward_acc = cond_df["forward_correct"] / cond_df["forward_total"]
            reverse_acc = cond_df["reverse_correct"] / cond_df["reverse_total"]

            # Bootstrap CIs
            _, f_ci_lower, f_ci_upper = bootstrap_ci(forward_acc.values)
            _, r_ci_lower, r_ci_upper = bootstrap_ci(reverse_acc.values)

            # Asymmetry
            asymmetry = forward_acc.mean() - reverse_acc.mean()
            asymmetry_values = forward_acc.values - reverse_acc.values
            _, a_ci_lower, a_ci_upper = bootstrap_ci(asymmetry_values)

            # Within-subjects test (McNemar's for paired proportions)
            forward_binary = (cond_df["forward_correct"] / cond_df["forward_total"] > 0.5).astype(int).values
            reverse_binary = (cond_df["reverse_correct"] / cond_df["reverse_total"] > 0.5).astype(int).values

            # Paired t-test on accuracy scores
            t_stat, p_value = stats.ttest_rel(forward_acc, reverse_acc)
            cohens_d = compute_cohens_d(forward_acc.values, reverse_acc.values)

            within_test = StatisticalResult(
                test_name="Paired t-test",
                statistic=t_stat,
                p_value=p_value,
                effect_size=cohens_d,
                effect_size_name="Cohen's d",
                ci_lower=a_ci_lower,
                ci_upper=a_ci_upper,
                df=len(cond_df) - 1,
                n=len(cond_df),
            )

            results[condition] = ConditionResults(
                condition=condition,
                n_participants=len(cond_df),
                forward_accuracy=forward_acc.mean(),
                forward_ci=(f_ci_lower, f_ci_upper),
                reverse_accuracy=reverse_acc.mean(),
                reverse_ci=(r_ci_lower, r_ci_upper),
                asymmetry=asymmetry,
                asymmetry_ci=(a_ci_lower, a_ci_upper),
                within_test=within_test,
            )

        return results

    def run_anova(self) -> Dict[str, StatisticalResult]:
        """
        Run the primary 2×3 mixed ANOVA.

        Returns
        -------
        Dict[str, StatisticalResult]
            ANOVA results for main effects and interaction
        """
        if self.clean_data is None:
            self.apply_exclusions()

        # Convert to long format for ANOVA
        df = self.clean_data.copy()

        # Create long-format data
        long_data = []
        for _, row in df.iterrows():
            long_data.append({
                "participant_id": row["participant_id"],
                "condition": row["condition"],
                "direction": "forward",
                "accuracy": row["forward_correct"] / row["forward_total"],
            })
            long_data.append({
                "participant_id": row["participant_id"],
                "condition": row["condition"],
                "direction": "reverse",
                "accuracy": row["reverse_correct"] / row["reverse_total"],
            })

        long_df = pd.DataFrame(long_data)

        # Run mixed ANOVA
        anova_results = anova_2x3_mixed(
            data=long_df,
            dv="accuracy",
            within="direction",
            between="condition",
            subject="participant_id"
        )

        return anova_results

    def test_asymmetry_flip(self) -> StatisticalResult:
        """
        Test whether asymmetry flips between A-then-B and B-then-A conditions.

        This is the critical test of the predictive memory hypothesis.

        Returns
        -------
        StatisticalResult
            Result of the flip test
        """
        if self.clean_data is None:
            self.apply_exclusions()

        df = self.clean_data

        # Get asymmetry scores for each condition
        a_then_b = df[df["condition"] == "A_then_B"]
        b_then_a = df[df["condition"] == "B_then_A"]

        asymmetry_ab = (
            a_then_b["forward_correct"] / a_then_b["forward_total"] -
            a_then_b["reverse_correct"] / a_then_b["reverse_total"]
        ).values

        asymmetry_ba = (
            b_then_a["forward_correct"] / b_then_a["forward_total"] -
            b_then_a["reverse_correct"] / b_then_a["reverse_total"]
        ).values

        # Independent samples t-test on asymmetry scores
        t_stat, p_value = stats.ttest_ind(asymmetry_ab, asymmetry_ba)
        cohens_d = compute_cohens_d(asymmetry_ab, asymmetry_ba)

        # Calculate flip magnitude
        flip_magnitude = np.mean(asymmetry_ab) - np.mean(asymmetry_ba)

        # Bootstrap CI for flip magnitude
        n_bootstrap = 10000
        rng = np.random.default_rng(42)
        bootstrap_flips = []

        for _ in range(n_bootstrap):
            ab_sample = rng.choice(asymmetry_ab, size=len(asymmetry_ab), replace=True)
            ba_sample = rng.choice(asymmetry_ba, size=len(asymmetry_ba), replace=True)
            bootstrap_flips.append(np.mean(ab_sample) - np.mean(ba_sample))

        ci_lower = np.percentile(bootstrap_flips, 2.5)
        ci_upper = np.percentile(bootstrap_flips, 97.5)

        return StatisticalResult(
            test_name="Asymmetry flip test (Independent t-test)",
            statistic=t_stat,
            p_value=p_value,
            effect_size=cohens_d,
            effect_size_name="Cohen's d",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            df=len(asymmetry_ab) + len(asymmetry_ba) - 2,
            n=len(asymmetry_ab) + len(asymmetry_ba),
            additional_info={
                "flip_magnitude": flip_magnitude,
                "mean_asymmetry_A_then_B": np.mean(asymmetry_ab),
                "mean_asymmetry_B_then_A": np.mean(asymmetry_ba),
            },
        )

    def test_simultaneous_equivalence(
        self,
        equivalence_bound: float = 0.1
    ) -> Dict[str, Any]:
        """
        Test whether simultaneous condition shows no asymmetry (equivalence test).

        Parameters
        ----------
        equivalence_bound : float
            Bound for practical equivalence (proportion difference)

        Returns
        -------
        Dict[str, Any]
            Equivalence test results
        """
        if self.clean_data is None:
            self.apply_exclusions()

        df = self.clean_data
        simul = df[df["condition"] == "simultaneous"]

        forward_acc = (simul["forward_correct"] / simul["forward_total"]).values
        reverse_acc = (simul["reverse_correct"] / simul["reverse_total"]).values

        result = equivalence_test(
            forward_acc,
            reverse_acc,
            equivalence_bound=equivalence_bound
        )

        return result

    def compute_effect_sizes(self) -> Dict[str, float]:
        """
        Compute all relevant effect sizes.

        Returns
        -------
        Dict[str, float]
            Named effect sizes
        """
        condition_results = self.compute_condition_results()
        effect_sizes = {}

        # Within-condition asymmetries (Cohen's h)
        for condition, results in condition_results.items():
            h = compute_cohens_h(
                results.forward_accuracy,
                results.reverse_accuracy
            )
            effect_sizes[f"cohens_h_{condition}"] = h

        # Overall reversal curse effect
        all_forward = []
        all_reverse = []

        for condition in ["A_then_B", "B_then_A"]:
            if condition in condition_results:
                all_forward.append(condition_results[condition].forward_accuracy)
                all_reverse.append(condition_results[condition].reverse_accuracy)

        if all_forward and all_reverse:
            effect_sizes["cohens_h_overall"] = compute_cohens_h(
                np.mean(all_forward),
                np.mean(all_reverse)
            )

        # Flip magnitude
        flip_test = self.test_asymmetry_flip()
        effect_sizes["cohens_d_flip"] = flip_test.effect_size
        effect_sizes["flip_magnitude_pp"] = flip_test.additional_info.get(
            "flip_magnitude", 0
        ) * 100  # Convert to percentage points

        return effect_sizes

    def run_full_analysis(self) -> ExperimentalResults:
        """
        Run the complete pre-registered analysis.

        Returns
        -------
        ExperimentalResults
            Complete analysis results
        """
        logger.info("Running full experimental analysis...")

        # Apply exclusions
        _, exclusion_reasons = self.apply_exclusions()

        # Condition-level results
        condition_results = self.compute_condition_results()

        # ANOVA
        anova_results = self.run_anova()

        # Flip test
        flip_test = self.test_asymmetry_flip()

        # Equivalence test
        equiv_test = self.test_simultaneous_equivalence()

        # Effect sizes
        effect_sizes = self.compute_effect_sizes()

        self.results = ExperimentalResults(
            n_total=len(self.raw_data) if self.raw_data is not None else 0,
            n_per_condition={
                cond: results.n_participants
                for cond, results in condition_results.items()
            },
            n_excluded=exclusion_reasons["total_excluded"],
            exclusion_reasons=exclusion_reasons,
            condition_results=condition_results,
            anova_results=anova_results,
            flip_test=flip_test,
            equivalence_test=equiv_test,
            effect_sizes=effect_sizes,
        )

        logger.info("Analysis complete.")
        return self.results

    def generate_results_table(self) -> pd.DataFrame:
        """
        Generate publication-ready results table.

        Returns
        -------
        pd.DataFrame
            Formatted results table
        """
        if self.results is None:
            self.run_full_analysis()

        rows = []
        for condition, results in self.results.condition_results.items():
            condition_label = {
                "A_then_B": "A-then-B",
                "B_then_A": "B-then-A",
                "simultaneous": "Simultaneous",
            }.get(condition, condition)

            rows.append({
                "Condition": condition_label,
                "N": results.n_participants,
                "Forward Acc": f"{results.forward_accuracy:.1%}",
                "Forward 95% CI": f"[{results.forward_ci[0]:.1%}, {results.forward_ci[1]:.1%}]",
                "Reverse Acc": f"{results.reverse_accuracy:.1%}",
                "Reverse 95% CI": f"[{results.reverse_ci[0]:.1%}, {results.reverse_ci[1]:.1%}]",
                "Asymmetry": f"{results.asymmetry:+.1%}",
                "p-value": f"{results.within_test.p_value:.4f}" if results.within_test.p_value >= 0.001 else "<.001",
            })

        return pd.DataFrame(rows)

    def generate_apa_summary(self) -> str:
        """
        Generate APA-formatted results summary.

        Returns
        -------
        str
            APA-formatted text
        """
        if self.results is None:
            self.run_full_analysis()

        sections = []

        # Sample description
        sections.append(
            f"Participants (N = {self.results.n_total}) were randomly assigned to "
            f"one of three conditions: A-then-B (n = {self.results.n_per_condition.get('A_then_B', 0)}), "
            f"B-then-A (n = {self.results.n_per_condition.get('B_then_A', 0)}), or "
            f"Simultaneous (n = {self.results.n_per_condition.get('simultaneous', 0)}). "
            f"After applying pre-registered exclusion criteria, "
            f"{self.results.n_excluded} participants were excluded."
        )

        # ANOVA results
        if "direction:condition" in self.results.anova_results:
            interaction = self.results.anova_results["direction:condition"]
            sections.append(
                f"\nA 2 × 3 mixed ANOVA revealed a significant Direction × Condition "
                f"interaction, {interaction.to_apa()}."
            )

        # Condition-level results
        for condition in ["A_then_B", "B_then_A", "simultaneous"]:
            if condition in self.results.condition_results:
                cr = self.results.condition_results[condition]
                condition_label = {
                    "A_then_B": "A-then-B",
                    "B_then_A": "B-then-A",
                    "simultaneous": "simultaneous",
                }[condition]

                sections.append(
                    f"\nIn the {condition_label} condition, participants showed "
                    f"{cr.forward_accuracy:.1%} accuracy on forward tests and "
                    f"{cr.reverse_accuracy:.1%} on reverse tests, "
                    f"{cr.within_test.to_apa()}."
                )

        # Flip test
        flip = self.results.flip_test
        sections.append(
            f"\nCritically, the asymmetry flipped between conditions: "
            f"A-then-B showed +{flip.additional_info['mean_asymmetry_A_then_B']:.1%} asymmetry "
            f"favoring forward, while B-then-A showed "
            f"{flip.additional_info['mean_asymmetry_B_then_A']:.1%} asymmetry favoring reverse. "
            f"This flip was highly significant, {flip.to_apa()}."
        )

        # Equivalence test
        if self.results.equivalence_test.get("equivalent", False):
            sections.append(
                f"\nThe simultaneous condition showed equivalent accuracy "
                f"in both directions (p = {self.results.equivalence_test['p_value']:.3f}), "
                f"supporting the hypothesis that sequential structure is necessary "
                f"for the asymmetry."
            )

        return "\n".join(sections)

    def export_results(self, output_dir: Path) -> None:
        """
        Export all results to files.

        Parameters
        ----------
        output_dir : Path
            Directory to save results
        """
        if self.results is None:
            self.run_full_analysis()

        output_dir.mkdir(parents=True, exist_ok=True)

        # Export clean data
        if self.clean_data is not None:
            self.clean_data.to_csv(
                output_dir / "clean_data.csv",
                index=False
            )

        # Export results table
        results_table = self.generate_results_table()
        results_table.to_csv(
            output_dir / "results_table.csv",
            index=False
        )

        # Export APA summary
        apa_summary = self.generate_apa_summary()
        with open(output_dir / "apa_summary.txt", "w") as f:
            f.write(apa_summary)

        # Export JSON summary
        summary = {
            "n_total": self.results.n_total,
            "n_per_condition": self.results.n_per_condition,
            "n_excluded": self.results.n_excluded,
            "exclusion_reasons": self.results.exclusion_reasons,
            "effect_sizes": self.results.effect_sizes,
            "flip_test": self.results.flip_test.to_dict(),
            "equivalence_test": self.results.equivalence_test,
            "condition_results": {
                cond: {
                    "n": cr.n_participants,
                    "forward_accuracy": cr.forward_accuracy,
                    "reverse_accuracy": cr.reverse_accuracy,
                    "asymmetry": cr.asymmetry,
                    "p_value": cr.within_test.p_value,
                }
                for cond, cr in self.results.condition_results.items()
            },
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Results exported to {output_dir}")


def run_experimental_analysis(
    data_path: Path,
    output_dir: Path
) -> ExperimentalResults:
    """
    Run the complete experimental analysis pipeline.

    Parameters
    ----------
    data_path : Path
        Path to experimental data file
    output_dir : Path
        Directory for output files

    Returns
    -------
    ExperimentalResults
        Complete analysis results
    """
    analyzer = ExperimentalAnalyzer()

    # Load data
    analyzer.load_data(data_path)

    # Run analysis
    results = analyzer.run_full_analysis()

    # Export results
    analyzer.export_results(output_dir)

    return results
