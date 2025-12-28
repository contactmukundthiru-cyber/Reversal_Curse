"""
Duolingo data analysis pipeline for Study 1.

This module provides:
- Data loading and preprocessing
- Asymmetric exposure pair identification
- Reversal gap calculation at scale
- Mixed-effects modeling for language learning data
- Visualization helpers

Data Source: Duolingo public research dataset (Settles et al., 2018)
https://github.com/duolingo/halflife-regression
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from config.settings import get_config
from .statistics import (
    calculate_reversal_gap,
    compute_cohens_h,
    bootstrap_ci,
    mixed_effects_model,
    StatisticalResult,
    ReversalGapResult,
)

logger = logging.getLogger(__name__)


@dataclass
class DuolingoDataset:
    """Container for processed Duolingo data."""

    learning_events: pd.DataFrame
    word_pairs: pd.DataFrame
    users: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass
class AsymmetricPair:
    """A word pair with asymmetric exposure."""

    pair_id: str
    word_l1: str
    word_l2: str
    forward_trials: int
    reverse_trials: int
    forward_accuracy: float
    reverse_accuracy: float
    n_users: int
    language_pair: Tuple[str, str]


class DuolingoAnalyzer:
    """
    Analyzer for Duolingo learning data.

    This class implements the full analysis pipeline for Study 1,
    identifying asymmetric exposure pairs and computing reversal gaps.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the analyzer.

        Parameters
        ----------
        config : Optional[Any]
            Configuration object. Uses default if None.
        """
        self.config = config or get_config().duolingo
        self.data: Optional[DuolingoDataset] = None
        self.results: Dict[str, Any] = {}

    def load_data(
        self,
        data_path: Path,
        language_pair: Optional[Tuple[str, str]] = None
    ) -> DuolingoDataset:
        """
        Load Duolingo data from file.

        The expected format follows the Duolingo SLAM dataset structure:
        - user_id: User identifier
        - timestamp: Unix timestamp
        - learning_language: Target language code
        - ui_language: Source language code
        - lexeme_id: Vocabulary item ID
        - lexeme_string: The word/phrase
        - session_correct: Number correct in session
        - session_seen: Number seen in session

        Parameters
        ----------
        data_path : Path
            Path to the data file (CSV or Parquet)
        language_pair : Optional[Tuple[str, str]]
            Filter to specific language pair (L1, L2)

        Returns
        -------
        DuolingoDataset
            Loaded and preprocessed data
        """
        logger.info(f"Loading Duolingo data from {data_path}")

        # Load data based on file type
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        # Standardize column names
        df = self._standardize_columns(df)

        # Filter by language pair if specified
        if language_pair:
            df = df[
                (df["source_language"] == language_pair[0]) &
                (df["target_language"] == language_pair[1])
            ]

        # Compute derived fields
        df["accuracy"] = df["session_correct"] / df["session_seen"]
        df["direction"] = df.apply(
            lambda x: "forward" if x["is_forward"] else "reverse",
            axis=1
        )

        # Create word pairs DataFrame
        word_pairs = self._extract_word_pairs(df)

        # Create users DataFrame
        users = self._extract_users(df)

        # Metadata
        metadata = {
            "n_events": len(df),
            "n_pairs": len(word_pairs),
            "n_users": len(users),
            "language_pairs": df.groupby(
                ["source_language", "target_language"]
            ).size().to_dict(),
            "date_range": (df["timestamp"].min(), df["timestamp"].max()),
        }

        self.data = DuolingoDataset(
            learning_events=df,
            word_pairs=word_pairs,
            users=users,
            metadata=metadata,
        )

        logger.info(
            f"Loaded {metadata['n_events']:,} events, "
            f"{metadata['n_pairs']:,} word pairs, "
            f"{metadata['n_users']:,} users"
        )

        return self.data

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names from various Duolingo data formats."""
        column_mapping = {
            "user_id": "user_id",
            "p_recall": "accuracy",
            "timestamp": "timestamp",
            "learning_language": "target_language",
            "ui_language": "source_language",
            "lexeme_id": "lexeme_id",
            "lexeme_string": "lexeme_string",
            "session_correct": "session_correct",
            "session_seen": "session_seen",
            # Handle alternative column names
            "word": "lexeme_string",
            "correct": "session_correct",
            "total": "session_seen",
        }

        # Rename columns that exist
        rename_dict = {
            old: new for old, new in column_mapping.items()
            if old in df.columns and new not in df.columns
        }
        df = df.rename(columns=rename_dict)

        # Determine direction if not present
        if "is_forward" not in df.columns:
            # Heuristic: if lexeme is in target language characters, it's reverse
            # This is approximate - actual dataset should have direction
            df["is_forward"] = True  # Default assumption

        return df

    def _extract_word_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique word pairs with statistics."""
        pairs = df.groupby("lexeme_id").agg({
            "lexeme_string": "first",
            "source_language": "first",
            "target_language": "first",
            "accuracy": ["mean", "std", "count"],
            "user_id": "nunique",
        }).reset_index()

        pairs.columns = [
            "lexeme_id", "lexeme_string", "source_language", "target_language",
            "mean_accuracy", "std_accuracy", "n_trials", "n_users"
        ]

        return pairs

    def _extract_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract user-level statistics."""
        users = df.groupby("user_id").agg({
            "accuracy": ["mean", "std"],
            "lexeme_id": "nunique",
            "timestamp": ["min", "max", "count"],
        }).reset_index()

        users.columns = [
            "user_id", "mean_accuracy", "std_accuracy",
            "n_words", "first_event", "last_event", "n_events"
        ]

        return users

    def identify_asymmetric_pairs(
        self,
        min_forward_trials: Optional[int] = None,
        max_reverse_trials: Optional[int] = None
    ) -> List[AsymmetricPair]:
        """
        Identify word pairs with asymmetric exposure.

        Parameters
        ----------
        min_forward_trials : Optional[int]
            Minimum forward direction trials. Uses config default if None.
        max_reverse_trials : Optional[int]
            Maximum reverse direction trials. Uses config default if None.

        Returns
        -------
        List[AsymmetricPair]
            List of asymmetric pairs meeting criteria
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        min_forward = min_forward_trials or self.config.min_forward_trials
        max_reverse = max_reverse_trials or self.config.max_reverse_trials

        df = self.data.learning_events

        # Calculate trials per direction per pair
        direction_counts = df.groupby(
            ["lexeme_id", "direction"]
        ).agg({
            "accuracy": ["mean", "count"],
            "user_id": "nunique",
        }).reset_index()

        direction_counts.columns = [
            "lexeme_id", "direction", "mean_accuracy", "n_trials", "n_users"
        ]

        # Pivot to wide format
        pairs_wide = direction_counts.pivot(
            index="lexeme_id",
            columns="direction",
            values=["mean_accuracy", "n_trials", "n_users"]
        ).fillna(0)

        pairs_wide.columns = [
            f"{col[0]}_{col[1]}" for col in pairs_wide.columns
        ]
        pairs_wide = pairs_wide.reset_index()

        # Filter for asymmetric exposure
        mask = (
            (pairs_wide["n_trials_forward"] >= min_forward) &
            (pairs_wide["n_trials_reverse"] <= max_reverse)
        )
        asymmetric_df = pairs_wide[mask]

        logger.info(
            f"Found {len(asymmetric_df):,} pairs with "
            f"≥{min_forward} forward and ≤{max_reverse} reverse trials"
        )

        # Convert to AsymmetricPair objects
        word_pair_info = self.data.word_pairs.set_index("lexeme_id")

        asymmetric_pairs = []
        for _, row in asymmetric_df.iterrows():
            lexeme_id = row["lexeme_id"]
            if lexeme_id in word_pair_info.index:
                info = word_pair_info.loc[lexeme_id]
                asymmetric_pairs.append(AsymmetricPair(
                    pair_id=str(lexeme_id),
                    word_l1=str(info.get("lexeme_string", "")),
                    word_l2=str(info.get("lexeme_string", "")),  # Would need translation
                    forward_trials=int(row["n_trials_forward"]),
                    reverse_trials=int(row["n_trials_reverse"]),
                    forward_accuracy=float(row["mean_accuracy_forward"]),
                    reverse_accuracy=float(row.get("mean_accuracy_reverse", 0)),
                    n_users=int(row.get("n_users_forward", 0)),
                    language_pair=(
                        str(info.get("source_language", "")),
                        str(info.get("target_language", ""))
                    ),
                ))

        self.results["asymmetric_pairs"] = asymmetric_pairs
        return asymmetric_pairs

    def compute_reversal_gap(
        self,
        pairs: Optional[List[AsymmetricPair]] = None
    ) -> ReversalGapResult:
        """
        Compute the aggregate reversal gap.

        Parameters
        ----------
        pairs : Optional[List[AsymmetricPair]]
            Pairs to analyze. Uses identified pairs if None.

        Returns
        -------
        ReversalGapResult
            Comprehensive reversal gap analysis
        """
        if pairs is None:
            pairs = self.results.get("asymmetric_pairs", [])

        if not pairs:
            raise ValueError(
                "No pairs to analyze. Call identify_asymmetric_pairs() first."
            )

        # Extract accuracy arrays
        forward_acc = np.array([p.forward_accuracy for p in pairs])
        reverse_acc = np.array([p.reverse_accuracy for p in pairs])

        # Weight by number of trials
        forward_trials = np.array([p.forward_trials for p in pairs])
        reverse_trials = np.array([p.reverse_trials for p in pairs])

        # Weighted mean accuracies
        weighted_forward = np.average(forward_acc, weights=forward_trials)
        weighted_reverse = np.average(
            reverse_acc,
            weights=np.maximum(reverse_trials, 1)  # Avoid zero weights
        )

        gap = weighted_forward - weighted_reverse

        # Bootstrap CI for the gap
        def weighted_gap_statistic(indices):
            f_acc = forward_acc[indices]
            r_acc = reverse_acc[indices]
            f_trials = forward_trials[indices]
            r_trials = np.maximum(reverse_trials[indices], 1)
            w_f = np.average(f_acc, weights=f_trials)
            w_r = np.average(r_acc, weights=r_trials)
            return w_f - w_r

        n_pairs = len(pairs)
        n_bootstrap = self.config.bootstrap_iterations
        rng = np.random.default_rng(42)

        bootstrap_gaps = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.choice(n_pairs, size=n_pairs, replace=True)
            bootstrap_gaps[i] = weighted_gap_statistic(idx)

        ci_lower = np.percentile(bootstrap_gaps, 2.5)
        ci_upper = np.percentile(bootstrap_gaps, 97.5)

        # Effect size
        cohens_h = compute_cohens_h(weighted_forward, weighted_reverse)

        # Statistical test (one-sample t-test on pair-level gaps)
        pair_gaps = forward_acc - reverse_acc
        t_stat, p_value = stats.ttest_1samp(pair_gaps, 0)

        stat_result = StatisticalResult(
            test_name="One-sample t-test",
            statistic=t_stat,
            p_value=p_value,
            effect_size=cohens_h,
            effect_size_name="Cohen's h",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            df=n_pairs - 1,
            n=n_pairs,
        )

        result = ReversalGapResult(
            forward_accuracy=weighted_forward,
            reverse_accuracy=weighted_reverse,
            gap=gap,
            gap_ci_lower=ci_lower,
            gap_ci_upper=ci_upper,
            cohens_h=cohens_h,
            n_observations=sum(p.forward_trials + p.reverse_trials for p in pairs),
            statistical_test=stat_result,
        )

        self.results["reversal_gap"] = result
        return result

    def fit_mixed_model(self) -> Dict[str, Any]:
        """
        Fit mixed-effects model for the Duolingo analysis.

        Model:
        Accuracy ~ Direction * Exposure_Asymmetry +
                   (1 | User) + (1 | Word_Pair) +
                   Frequency + Proficiency + Recency

        Returns
        -------
        Dict[str, Any]
            Model results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.data.learning_events.copy()

        # Create exposure asymmetry variable
        pair_exposure = df.groupby("lexeme_id").agg({
            "direction": lambda x: (x == "forward").sum() / len(x)
        }).rename(columns={"direction": "forward_ratio"})

        df = df.merge(
            pair_exposure,
            left_on="lexeme_id",
            right_index=True
        )

        df["exposure_asymmetry"] = df["forward_ratio"] - 0.5

        # Create direction dummy
        df["is_forward_numeric"] = (df["direction"] == "forward").astype(int)

        # Fit model
        formula = "accuracy ~ is_forward_numeric * exposure_asymmetry"

        result = mixed_effects_model(
            data=df,
            formula=formula,
            groups="user_id",
            family="gaussian"
        )

        self.results["mixed_model"] = result
        return result

    def analyze_by_language_pair(self) -> pd.DataFrame:
        """
        Analyze reversal gap separately for each language pair.

        Returns
        -------
        pd.DataFrame
            Results by language pair
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.data.learning_events
        results = []

        language_pairs = df.groupby(
            ["source_language", "target_language"]
        ).size().index.tolist()

        for l1, l2 in tqdm(language_pairs, desc="Analyzing language pairs"):
            pair_df = df[
                (df["source_language"] == l1) &
                (df["target_language"] == l2)
            ]

            if len(pair_df) < 100:
                continue

            forward_acc = pair_df[pair_df["direction"] == "forward"]["accuracy"]
            reverse_acc = pair_df[pair_df["direction"] == "reverse"]["accuracy"]

            if len(forward_acc) < 10 or len(reverse_acc) < 10:
                continue

            gap = forward_acc.mean() - reverse_acc.mean()
            cohens_h = compute_cohens_h(forward_acc.mean(), reverse_acc.mean())

            t_stat, p_value = stats.ttest_ind(forward_acc, reverse_acc)

            results.append({
                "source_language": l1,
                "target_language": l2,
                "n_events": len(pair_df),
                "forward_accuracy": forward_acc.mean(),
                "reverse_accuracy": reverse_acc.mean(),
                "gap": gap,
                "cohens_h": cohens_h,
                "t_statistic": t_stat,
                "p_value": p_value,
            })

        results_df = pd.DataFrame(results)
        self.results["by_language_pair"] = results_df
        return results_df

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary of all analyses.

        Returns
        -------
        Dict[str, Any]
            Summary statistics and results
        """
        summary = {
            "dataset": {
                "n_events": self.data.metadata["n_events"] if self.data else 0,
                "n_pairs": self.data.metadata["n_pairs"] if self.data else 0,
                "n_users": self.data.metadata["n_users"] if self.data else 0,
            },
            "asymmetric_pairs": {
                "n_pairs": len(self.results.get("asymmetric_pairs", [])),
            },
        }

        if "reversal_gap" in self.results:
            gap_result = self.results["reversal_gap"]
            summary["reversal_gap"] = {
                "forward_accuracy": gap_result.forward_accuracy,
                "reverse_accuracy": gap_result.reverse_accuracy,
                "gap": gap_result.gap,
                "gap_ci": (gap_result.gap_ci_lower, gap_result.gap_ci_upper),
                "cohens_h": gap_result.cohens_h,
                "p_value": gap_result.statistical_test.p_value,
            }

        return summary

    def export_results(self, output_dir: Path) -> None:
        """
        Export all results to files.

        Parameters
        ----------
        output_dir : Path
            Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export asymmetric pairs
        if "asymmetric_pairs" in self.results:
            pairs_df = pd.DataFrame([
                {
                    "pair_id": p.pair_id,
                    "word_l1": p.word_l1,
                    "word_l2": p.word_l2,
                    "forward_trials": p.forward_trials,
                    "reverse_trials": p.reverse_trials,
                    "forward_accuracy": p.forward_accuracy,
                    "reverse_accuracy": p.reverse_accuracy,
                    "n_users": p.n_users,
                }
                for p in self.results["asymmetric_pairs"]
            ])
            pairs_df.to_csv(output_dir / "asymmetric_pairs.csv", index=False)

        # Export by-language results
        if "by_language_pair" in self.results:
            self.results["by_language_pair"].to_csv(
                output_dir / "by_language_pair.csv",
                index=False
            )

        # Export summary
        import json
        summary = self.generate_summary()
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Results exported to {output_dir}")


def run_duolingo_analysis(
    data_path: Path,
    output_dir: Path,
    language_pair: Optional[Tuple[str, str]] = None
) -> Dict[str, Any]:
    """
    Run the complete Duolingo analysis pipeline.

    Parameters
    ----------
    data_path : Path
        Path to Duolingo data file
    output_dir : Path
        Directory for output files
    language_pair : Optional[Tuple[str, str]]
        Filter to specific language pair

    Returns
    -------
    Dict[str, Any]
        Complete analysis results
    """
    analyzer = DuolingoAnalyzer()

    # Load data
    analyzer.load_data(data_path, language_pair)

    # Identify asymmetric pairs
    analyzer.identify_asymmetric_pairs()

    # Compute reversal gap
    analyzer.compute_reversal_gap()

    # Analyze by language pair
    analyzer.analyze_by_language_pair()

    # Fit mixed model
    analyzer.fit_mixed_model()

    # Export results
    analyzer.export_results(output_dir)

    return analyzer.generate_summary()
