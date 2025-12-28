"""
Configuration settings for the Temporal Directionality research project.

This module contains all configurable parameters for the research study,
including experimental parameters, analysis settings, and system configuration.

Note: This project was previously named "Reversal Curse" - rebranded to emphasize
the fundamental constraint on knowledge encoding rather than the AI comparison.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
EXPORTS_DIR = DATA_DIR / "exports"


@dataclass
class ExperimentConfig:
    """Configuration for the controlled experiment (Study 3).

    CRITICAL DESIGN NOTES:
    - Both test directions use 4-AFC (matched format to eliminate recall/recognition confound)
    - Manipulation check replaces criterion check (applied symmetrically)
    - Simultaneous condition includes bidirectional probes to ensure encoding
    """

    # Participant settings
    target_n_per_condition: int = 60
    target_n_after_exclusions: int = 50
    min_age: int = 18
    max_age: int = 45
    payment_usd: float = 2.50
    estimated_duration_minutes: int = 14  # Updated for longer test phase

    # Stimulus settings
    n_symbol_label_pairs: int = 16
    n_manipulation_check_pairs: int = 4  # Excluded from main analysis
    n_test_pairs: int = 12  # 16 - 4 manipulation check pairs
    label_min_length: int = 5
    label_max_length: int = 7

    # Training settings
    training_repetitions: int = 6
    fixation_duration_ms: int = 500
    stimulus_duration_ms: int = 1500
    blank_duration_ms: int = 500
    simultaneous_duration_ms: int = 2000  # Reduced to allow for probes
    simultaneous_probe_duration_ms: int = 1500
    iti_duration_ms: int = 1000

    # Simultaneous condition bidirectional probes
    simultaneous_probe_rate: float = 0.50  # 50% of trials have probes
    simultaneous_forward_probe_rate: float = 0.25  # "Which label?"
    simultaneous_reverse_probe_rate: float = 0.25  # "Which symbol?"

    # Manipulation check (replaces criterion check)
    # Tests BOTH directions symmetrically; no direction-biased re-training
    manipulation_check_n_forward: int = 2  # Random pairs tested A→B
    manipulation_check_n_reverse: int = 2  # Random pairs tested B→A
    manipulation_check_threshold: float = 0.50  # Exclude if below (suggests inattention)

    # Distractor task
    distractor_duration_seconds: int = 60

    # Test settings - MATCHED FORMAT CRITICAL
    # Block 1: 4-AFC both directions (CONFIRMATORY)
    test_4afc_response_deadline_ms: int = 8000
    n_foil_options: int = 3  # For 4-AFC (1 target + 3 foils)

    # Block 2: Typed recall both directions (EXPLORATORY)
    test_recall_response_deadline_ms: int = 15000

    # Exclusion criteria (applied symmetrically)
    min_completion_time_minutes: int = 7  # Updated for longer test
    min_overall_4afc_accuracy: float = 0.30  # Exclude near-chance responding
    # NOTE: No direction-specific exclusion to avoid biasing effect estimate

    # Conditions
    conditions: list = field(default_factory=lambda: [
        "A_then_B",
        "B_then_A",
        "simultaneous"
    ])


@dataclass
class DuolingoConfig:
    """Configuration for Duolingo data analysis (Study 1)."""

    # Data filtering
    min_forward_trials: int = 10
    max_reverse_trials: int = 2
    min_user_events: int = 100

    # Analysis settings
    bootstrap_iterations: int = 10000
    confidence_level: float = 0.95

    # Language pairs to analyze
    language_pairs: list = field(default_factory=lambda: [
        ("en", "es"),  # English-Spanish
        ("en", "fr"),  # English-French
        ("en", "de"),  # English-German
        ("en", "pt"),  # English-Portuguese
        ("en", "it"),  # English-Italian
    ])


@dataclass
class WikipediaConfig:
    """Configuration for Wikipedia/Wikidata analysis (Study 2)."""

    # Relationship types to analyze
    relationship_types: list = field(default_factory=lambda: [
        {"name": "country_capital", "forward": "Country → Capital", "reverse": "Capital → Country"},
        {"name": "element_symbol", "forward": "Element → Symbol", "reverse": "Symbol → Element"},
        {"name": "person_birthyear", "forward": "Person → Birth Year", "reverse": "Birth Year → Person"},
        {"name": "inventor_invention", "forward": "Inventor → Invention", "reverse": "Invention → Inventor"},
        {"name": "word_synonym", "forward": "Word → Synonym", "reverse": "Synonym → Word"},
    ])

    # Sample sizes
    pairs_per_relationship: int = 500

    # Wikidata settings
    wikidata_endpoint: str = "https://query.wikidata.org/sparql"
    request_timeout: int = 60


@dataclass
class StatisticsConfig:
    """Configuration for statistical analysis."""

    # Significance thresholds
    alpha: float = 0.05
    equivalence_bound: float = 0.1  # For TOST equivalence testing

    # Effect size thresholds (Cohen's conventions)
    small_effect_h: float = 0.2
    medium_effect_h: float = 0.5
    large_effect_h: float = 0.8

    # Power analysis
    target_power: float = 0.95

    # Mixed effects model
    optimizer: str = "bobyqa"
    max_iterations: int = 10000


@dataclass
class VisualizationConfig:
    """Configuration for figure generation."""

    # Figure dimensions (Nature style)
    single_column_width: float = 3.5  # inches
    double_column_width: float = 7.0  # inches
    max_height: float = 9.0  # inches

    # Colors (colorblind-friendly palette)
    colors: dict = field(default_factory=lambda: {
        "forward": "#2166AC",      # Blue
        "reverse": "#B2182B",      # Red
        "simultaneous": "#4DAF4A", # Green
        "neutral": "#666666",      # Gray
        "highlight": "#FF7F00",    # Orange
    })

    # Typography
    font_family: str = "Arial"
    title_size: int = 10
    label_size: int = 9
    tick_size: int = 8
    legend_size: int = 8

    # Output formats
    output_formats: list = field(default_factory=lambda: ["pdf", "png", "svg"])
    dpi: int = 300


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""

    # SQLite (default for development)
    sqlite_path: Path = field(default_factory=lambda: DATA_DIR / "reversal_curse.db")

    # PostgreSQL (for production)
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "reversal_curse"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", ""))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))

    @property
    def sqlite_uri(self) -> str:
        return f"sqlite:///{self.sqlite_path}"

    @property
    def postgres_uri(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@dataclass
class ProlificConfig:
    """Configuration for Prolific integration."""

    api_token: str = field(default_factory=lambda: os.getenv("PROLIFIC_API_TOKEN", ""))
    api_base_url: str = "https://api.prolific.co/api/v1"

    # Study settings
    study_title: str = "Memory and Learning Study"
    study_description: str = (
        "A study investigating how people learn and remember associations "
        "between symbols and labels. Takes approximately 12 minutes."
    )

    # Eligibility
    eligible_countries: list = field(default_factory=lambda: ["US", "GB", "CA", "AU"])
    min_approval_rate: int = 95
    min_previous_submissions: int = 10


def _get_secret_key() -> str:
    """
    Get the secret key from environment with proper security handling.

    In development: generates a random key if not set (with warning)
    In production: REQUIRES the SECRET_KEY to be set explicitly
    """
    import secrets
    import warnings

    secret_key = os.getenv("SECRET_KEY")
    env = os.getenv("FLASK_ENV", "development")

    if secret_key:
        # Secret key is set - use it
        return secret_key

    if env == "production":
        # Production MUST have a secret key
        raise ValueError(
            "SECRET_KEY environment variable is required in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )

    # Development mode - generate a temporary key with warning
    warnings.warn(
        "SECRET_KEY not set. Generating temporary key for development. "
        "Set SECRET_KEY in .env for session persistence across restarts.",
        RuntimeWarning
    )
    return secrets.token_hex(32)


def _get_dashboard_password() -> str:
    """
    Get the dashboard password from environment with security handling.

    Requires explicit password in production.
    """
    import secrets
    import warnings

    password = os.getenv("DASHBOARD_PASSWORD")
    env = os.getenv("FLASK_ENV", "development")

    if password:
        return password

    if env == "production":
        raise ValueError(
            "DASHBOARD_PASSWORD environment variable is required in production. "
            "Set a secure password in your .env file."
        )

    # Development mode - generate temporary password
    temp_password = secrets.token_urlsafe(16)
    warnings.warn(
        f"DASHBOARD_PASSWORD not set. Using temporary password: {temp_password}",
        RuntimeWarning
    )
    return temp_password


@dataclass
class AppConfig:
    """Main application configuration."""

    # Environment
    env: str = field(default_factory=lambda: os.getenv("FLASK_ENV", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("FLASK_DEBUG", "true").lower() == "true")

    # Server settings
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "5000")))

    # Security
    secret_key: str = field(default_factory=_get_secret_key)
    dashboard_password: str = field(default_factory=_get_dashboard_password)

    # Session
    session_lifetime_hours: int = 24

    # Sub-configurations
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    duolingo: DuolingoConfig = field(default_factory=DuolingoConfig)
    wikipedia: WikipediaConfig = field(default_factory=WikipediaConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    prolific: ProlificConfig = field(default_factory=ProlificConfig)


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> AppConfig:
    """Reload configuration from environment."""
    global config
    load_dotenv(override=True)
    config = AppConfig()
    return config
