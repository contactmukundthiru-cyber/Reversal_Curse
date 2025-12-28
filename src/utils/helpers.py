"""
Utility helper functions for the Reversal Curse research project.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : Union[str, Path]
        Directory path

    Returns
    -------
    Path
        The directory path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to JSON file

    Returns
    -------
    Dict[str, Any]
        Loaded data
    """
    with open(path, "r") as f:
        return json.load(f)


def save_json(
    data: Dict[str, Any],
    path: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Save data to JSON file.

    Parameters
    ----------
    data : Dict[str, Any]
        Data to save
    path : Union[str, Path]
        Output path
    indent : int
        JSON indentation
    """
    path = Path(path)
    ensure_directory(path.parent)

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)

    logger.info(f"Saved JSON to {path}")


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format a decimal value as a percentage string.

    Parameters
    ----------
    value : float
        Value between 0 and 1
    decimal_places : int
        Number of decimal places

    Returns
    -------
    str
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"


def format_pvalue(p: float) -> str:
    """
    Format p-value according to APA guidelines.

    Parameters
    ----------
    p : float
        P-value

    Returns
    -------
    str
        Formatted p-value string
    """
    if p < 0.001:
        return "p < .001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def get_timestamp() -> str:
    """
    Get current timestamp string.

    Returns
    -------
    str
        Timestamp in YYYYMMDD_HHMMSS format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_duration(start: datetime, end: datetime) -> float:
    """
    Calculate duration in seconds between two datetime objects.

    Parameters
    ----------
    start : datetime
        Start time
    end : datetime
        End time

    Returns
    -------
    float
        Duration in seconds
    """
    return (end - start).total_seconds()


def chunk_list(lst: list, chunk_size: int) -> list:
    """
    Split a list into chunks of specified size.

    Parameters
    ----------
    lst : list
        List to split
    chunk_size : int
        Size of each chunk

    Returns
    -------
    list
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Parameters
    ----------
    numerator : float
        Numerator
    denominator : float
        Denominator
    default : float
        Default value if denominator is zero

    Returns
    -------
    float
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator
