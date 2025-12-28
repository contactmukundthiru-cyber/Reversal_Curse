"""
Data preprocessing utilities for the Reversal Curse research.

This module provides functions for:
- Duolingo data cleaning and transformation
- Wikipedia/Wikidata data preprocessing
- Experimental data validation and cleaning
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def preprocess_duolingo_data(
    raw_data: pd.DataFrame,
    min_user_events: int = 100,
    min_word_occurrences: int = 10,
    remove_outliers: bool = True,
    outlier_std: float = 3.0
) -> pd.DataFrame:
    """
    Preprocess Duolingo learning data.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw Duolingo data
    min_user_events : int
        Minimum events per user to include
    min_word_occurrences : int
        Minimum occurrences per word
    remove_outliers : bool
        Whether to remove statistical outliers
    outlier_std : float
        Standard deviations for outlier threshold

    Returns
    -------
    pd.DataFrame
        Preprocessed data
    """
    logger.info(f"Preprocessing {len(raw_data):,} Duolingo records")
    df = raw_data.copy()

    # Standardize column names
    column_mapping = {
        "user_id": "user_id",
        "p_recall": "accuracy",
        "timestamp": "timestamp",
        "learning_language": "target_language",
        "ui_language": "source_language",
        "lexeme_id": "lexeme_id",
        "lexeme_string": "lexeme_string",
    }

    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Remove users with too few events
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_events].index
    df = df[df["user_id"].isin(valid_users)]
    logger.info(f"After user filter: {len(df):,} records")

    # Remove words with too few occurrences
    word_counts = df["lexeme_id"].value_counts()
    valid_words = word_counts[word_counts >= min_word_occurrences].index
    df = df[df["lexeme_id"].isin(valid_words)]
    logger.info(f"After word filter: {len(df):,} records")

    # Ensure accuracy is in valid range
    if "accuracy" in df.columns:
        df["accuracy"] = df["accuracy"].clip(0, 1)

    # Remove outliers in response time if present
    if remove_outliers and "response_time" in df.columns:
        mean_rt = df["response_time"].mean()
        std_rt = df["response_time"].std()
        lower_bound = mean_rt - outlier_std * std_rt
        upper_bound = mean_rt + outlier_std * std_rt
        df = df[
            (df["response_time"] >= lower_bound) &
            (df["response_time"] <= upper_bound)
        ]
        logger.info(f"After outlier removal: {len(df):,} records")

    # Add derived columns
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date

    # Sort by timestamp
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    logger.info(f"Preprocessing complete: {len(df):,} records")
    return df


def preprocess_wikipedia_data(
    fact_pairs: pd.DataFrame,
    remove_duplicates: bool = True,
    validate_pairs: bool = True,
    min_label_length: int = 2
) -> pd.DataFrame:
    """
    Preprocess Wikipedia/Wikidata fact pairs.

    Parameters
    ----------
    fact_pairs : pd.DataFrame
        Raw fact pair data
    remove_duplicates : bool
        Whether to remove duplicate pairs
    validate_pairs : bool
        Whether to validate pair structure
    min_label_length : int
        Minimum length for labels

    Returns
    -------
    pd.DataFrame
        Preprocessed fact pairs
    """
    logger.info(f"Preprocessing {len(fact_pairs):,} fact pairs")
    df = fact_pairs.copy()

    # Remove missing values
    required_cols = ["entity_a_label", "entity_b_label"]
    for col in required_cols:
        if col in df.columns:
            df = df[df[col].notna()]
            df = df[df[col].str.len() >= min_label_length]

    logger.info(f"After missing value removal: {len(df):,} pairs")

    # Remove duplicates
    if remove_duplicates:
        # Consider pairs as duplicates if A and B are the same
        if "entity_a_label" in df.columns and "entity_b_label" in df.columns:
            df = df.drop_duplicates(
                subset=["entity_a_label", "entity_b_label"]
            )
            logger.info(f"After duplicate removal: {len(df):,} pairs")

    # Validate pairs
    if validate_pairs:
        # Remove pairs where A == B
        if "entity_a_label" in df.columns and "entity_b_label" in df.columns:
            df = df[df["entity_a_label"] != df["entity_b_label"]]
            logger.info(f"After validation: {len(df):,} pairs")

    # Clean labels
    for col in ["entity_a_label", "entity_b_label"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    logger.info(f"Preprocessing complete: {len(df):,} pairs")
    return df


def validate_experimental_data(
    data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validate and clean experimental data.

    Applies pre-registered exclusion criteria and data quality checks.

    Parameters
    ----------
    data : pd.DataFrame
        Raw experimental data
    config : Optional[Dict[str, Any]]
        Validation configuration

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        (cleaned_data, exclusion_counts)
    """
    if config is None:
        config = {
            "min_completion_time_minutes": 5,
            "min_trained_accuracy": 0.50,
            "require_attention_check": True,
        }

    logger.info(f"Validating {len(data):,} experimental records")
    df = data.copy()

    exclusions = {
        "attention_check": 0,
        "low_accuracy": 0,
        "too_fast": 0,
        "incomplete": 0,
        "duplicate": 0,
        "total": 0,
    }

    initial_n = len(df)

    # Check for required columns
    required_cols = [
        "participant_id",
        "condition",
        "forward_correct",
        "forward_total",
        "reverse_correct",
        "reverse_total",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate derived metrics
    df["forward_accuracy"] = df["forward_correct"] / df["forward_total"]
    df["reverse_accuracy"] = df["reverse_correct"] / df["reverse_total"]

    # Trained direction accuracy
    df["trained_accuracy"] = df.apply(
        lambda row: (
            row["forward_accuracy"]
            if row["condition"] == "A_then_B"
            else row["reverse_accuracy"]
            if row["condition"] == "B_then_A"
            else (row["forward_accuracy"] + row["reverse_accuracy"]) / 2
        ),
        axis=1
    )

    # Remove duplicates
    if "prolific_pid" in df.columns:
        duplicate_mask = df.duplicated(subset=["prolific_pid"], keep="first")
        exclusions["duplicate"] = duplicate_mask.sum()
        df = df[~duplicate_mask]

    # Attention check
    if config.get("require_attention_check") and "attention_check_passed" in df.columns:
        attention_mask = df["attention_check_passed"] == True
        exclusions["attention_check"] = (~attention_mask).sum()
    else:
        attention_mask = pd.Series([True] * len(df), index=df.index)

    # Accuracy threshold
    accuracy_mask = df["trained_accuracy"] >= config.get("min_trained_accuracy", 0.5)
    exclusions["low_accuracy"] = (attention_mask & ~accuracy_mask).sum()

    # Completion time
    min_time = config.get("min_completion_time_minutes", 5) * 60
    if "completion_time" in df.columns:
        time_mask = df["completion_time"] >= min_time
        exclusions["too_fast"] = (attention_mask & accuracy_mask & ~time_mask).sum()
    else:
        time_mask = pd.Series([True] * len(df), index=df.index)

    # Completion status
    if "completed" in df.columns:
        completion_mask = df["completed"] == True
        exclusions["incomplete"] = (
            attention_mask & accuracy_mask & time_mask & ~completion_mask
        ).sum()
    else:
        completion_mask = pd.Series([True] * len(df), index=df.index)

    # Apply all exclusions
    keep_mask = attention_mask & accuracy_mask & time_mask & completion_mask
    df_clean = df[keep_mask].copy()

    exclusions["total"] = initial_n - len(df_clean)

    logger.info(
        f"Validation complete: {len(df_clean):,} records retained "
        f"({exclusions['total']} excluded)"
    )

    return df_clean, exclusions


def compute_summary_statistics(
    data: pd.DataFrame,
    groupby: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute summary statistics for experimental data.

    Parameters
    ----------
    data : pd.DataFrame
        Experimental data
    groupby : Optional[str]
        Column to group by (e.g., "condition")

    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    numeric_cols = [
        "forward_accuracy",
        "reverse_accuracy",
        "trained_accuracy",
        "completion_time",
        "trials_to_criterion",
    ]

    available_cols = [c for c in numeric_cols if c in data.columns]

    if groupby and groupby in data.columns:
        summary = data.groupby(groupby)[available_cols].agg([
            "count", "mean", "std", "min", "max",
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
        ])
    else:
        summary = data[available_cols].agg([
            "count", "mean", "std", "min", "max",
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
        ])

    return summary


def create_long_format(
    data: pd.DataFrame,
    id_vars: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert wide-format experimental data to long format for ANOVA.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format data with forward_* and reverse_* columns
    id_vars : Optional[List[str]]
        ID variables to preserve

    Returns
    -------
    pd.DataFrame
        Long-format data with direction column
    """
    if id_vars is None:
        id_vars = ["participant_id", "condition"]

    available_id_vars = [c for c in id_vars if c in data.columns]

    long_data = []

    for _, row in data.iterrows():
        base_info = {col: row[col] for col in available_id_vars}

        # Forward trial
        long_data.append({
            **base_info,
            "direction": "forward",
            "accuracy": row.get("forward_accuracy", row.get("forward_correct", 0) / row.get("forward_total", 1)),
            "correct": row.get("forward_correct", 0),
            "total": row.get("forward_total", 0),
        })

        # Reverse trial
        long_data.append({
            **base_info,
            "direction": "reverse",
            "accuracy": row.get("reverse_accuracy", row.get("reverse_correct", 0) / row.get("reverse_total", 1)),
            "correct": row.get("reverse_correct", 0),
            "total": row.get("reverse_total", 0),
        })

    return pd.DataFrame(long_data)


def export_for_analysis(
    data: pd.DataFrame,
    output_path: Path,
    format: str = "csv"
) -> None:
    """
    Export data for external analysis tools.

    Parameters
    ----------
    data : pd.DataFrame
        Data to export
    output_path : Path
        Output file path
    format : str
        Output format ("csv", "parquet", "json", "stata", "spss")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        data.to_csv(output_path, index=False)
    elif format == "parquet":
        data.to_parquet(output_path, index=False)
    elif format == "json":
        data.to_json(output_path, orient="records", indent=2)
    elif format == "stata":
        data.to_stata(output_path)
    elif format == "excel":
        data.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Exported {len(data)} records to {output_path}")
