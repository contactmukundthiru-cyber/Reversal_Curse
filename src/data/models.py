"""
Database models for the Reversal Curse experimental platform.

This module defines SQLAlchemy models for:
- Participants: Demographic and session information
- Trials: Individual trial data
- ExperimentSessions: Complete session records
- Stimuli: Symbol-label pairs
- StudyConfiguration: Experiment parameters
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import json

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.orm import relationship

db = SQLAlchemy()


class TimestampMixin:
    """Mixin for created/updated timestamps."""

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )


class StudyConfiguration(db.Model, TimestampMixin):
    """
    Configuration for an experiment deployment.

    Stores all parameters needed to run a specific version
    of the experiment.
    """

    __tablename__ = "study_configurations"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=False)

    # Experiment parameters
    n_symbol_label_pairs = db.Column(db.Integer, default=16)
    n_manipulation_check_pairs = db.Column(db.Integer, default=4)
    training_repetitions = db.Column(db.Integer, default=6)
    fixation_duration_ms = db.Column(db.Integer, default=500)
    stimulus_duration_ms = db.Column(db.Integer, default=1500)
    blank_duration_ms = db.Column(db.Integer, default=500)
    simultaneous_duration_ms = db.Column(db.Integer, default=2000)
    iti_duration_ms = db.Column(db.Integer, default=1000)

    # ISI manipulation (Study 3b) - interstimulus interval between A and B
    isi_duration_ms = db.Column(db.Integer, default=500)
    isi_conditions_json = db.Column(
        db.Text,
        default='[100, 500, 2000]'  # Short, medium, long ISI
    )

    # Test settings
    test_4afc_deadline_ms = db.Column(db.Integer, default=8000)
    test_recall_deadline_ms = db.Column(db.Integer, default=15000)

    # Manipulation check (replaces criterion check)
    manipulation_check_n_forward = db.Column(db.Integer, default=2)
    manipulation_check_n_reverse = db.Column(db.Integer, default=2)
    manipulation_check_threshold = db.Column(db.Float, default=0.50)

    # Simultaneous condition bidirectional probes
    simultaneous_probe_rate = db.Column(db.Float, default=0.50)

    # Phase 5 fast retraining (Study 3c)
    enable_phase5_retraining = db.Column(db.Boolean, default=False)
    phase5_retraining_reps = db.Column(db.Integer, default=3)

    # Block 2 typed recall (exploratory)
    enable_typed_recall = db.Column(db.Boolean, default=True)

    # Condition allocation
    conditions_json = db.Column(
        db.Text,
        default='["A_then_B", "B_then_A", "simultaneous"]'
    )

    # Target sample sizes
    target_n_per_condition = db.Column(db.Integer, default=60)

    # Prolific settings
    prolific_study_id = db.Column(db.String(100))
    payment_usd = db.Column(db.Float, default=2.50)

    # Relationships
    sessions = relationship(
        "ExperimentSession",
        back_populates="configuration"
    )

    @property
    def conditions(self) -> List[str]:
        """Get conditions as list."""
        return json.loads(self.conditions_json)

    @conditions.setter
    def conditions(self, value: List[str]) -> None:
        """Set conditions from list."""
        self.conditions_json = json.dumps(value)

    @property
    def isi_conditions(self) -> List[int]:
        """Get ISI conditions as list."""
        if self.isi_conditions_json:
            return json.loads(self.isi_conditions_json)
        return [100, 500, 2000]

    @isi_conditions.setter
    def isi_conditions(self, value: List[int]) -> None:
        """Set ISI conditions from list."""
        self.isi_conditions_json = json.dumps(value)

    def get_condition_counts(self) -> Dict[str, int]:
        """Get current participant counts per condition."""
        from sqlalchemy import func

        counts = (
            db.session.query(
                ExperimentSession.condition,
                func.count(ExperimentSession.id)
            )
            .filter(ExperimentSession.configuration_id == self.id)
            .filter(ExperimentSession.completed == True)
            .group_by(ExperimentSession.condition)
            .all()
        )
        return {cond: count for cond, count in counts}

    def allocate_condition(self) -> str:
        """Allocate a participant to a condition (balanced assignment)."""
        counts = self.get_condition_counts()
        # Find condition with fewest participants
        min_count = min(counts.get(c, 0) for c in self.conditions)
        for condition in self.conditions:
            if counts.get(condition, 0) == min_count:
                return condition
        return self.conditions[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "n_symbol_label_pairs": self.n_symbol_label_pairs,
            "n_manipulation_check_pairs": self.n_manipulation_check_pairs,
            "training_repetitions": self.training_repetitions,
            "fixation_duration_ms": self.fixation_duration_ms,
            "stimulus_duration_ms": self.stimulus_duration_ms,
            "blank_duration_ms": self.blank_duration_ms,
            "simultaneous_duration_ms": self.simultaneous_duration_ms,
            "iti_duration_ms": self.iti_duration_ms,
            "isi_duration_ms": self.isi_duration_ms,
            "isi_conditions": self.isi_conditions,
            "test_4afc_deadline_ms": self.test_4afc_deadline_ms,
            "test_recall_deadline_ms": self.test_recall_deadline_ms,
            "manipulation_check_n_forward": self.manipulation_check_n_forward,
            "manipulation_check_n_reverse": self.manipulation_check_n_reverse,
            "manipulation_check_threshold": self.manipulation_check_threshold,
            "simultaneous_probe_rate": self.simultaneous_probe_rate,
            "enable_phase5_retraining": self.enable_phase5_retraining,
            "phase5_retraining_reps": self.phase5_retraining_reps,
            "enable_typed_recall": self.enable_typed_recall,
            "conditions": self.conditions,
            "target_n_per_condition": self.target_n_per_condition,
            "condition_counts": self.get_condition_counts(),
        }


class Stimulus(db.Model, TimestampMixin):
    """
    A symbol-label pair stimulus.

    Symbols are stored as SVG paths or image references.
    Labels are pronounceable nonwords.
    """

    __tablename__ = "stimuli"

    id = db.Column(db.Integer, primary_key=True)
    stimulus_set_id = db.Column(db.String(50), nullable=False, index=True)

    # Symbol properties
    symbol_id = db.Column(db.String(50), nullable=False)
    symbol_svg = db.Column(db.Text)  # SVG path data
    symbol_image_path = db.Column(db.String(255))

    # Label properties
    label = db.Column(db.String(20), nullable=False)
    label_phonetic = db.Column(db.String(50))

    # Validation flags
    is_validated = db.Column(db.Boolean, default=False)
    validation_notes = db.Column(db.Text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "stimulus_set_id": self.stimulus_set_id,
            "symbol_id": self.symbol_id,
            "symbol_svg": self.symbol_svg,
            "symbol_image_path": self.symbol_image_path,
            "label": self.label,
            "label_phonetic": self.label_phonetic,
        }


class Participant(db.Model, TimestampMixin):
    """
    Participant demographic and meta information.

    Stores information collected from Prolific and exit surveys.
    """

    __tablename__ = "participants"

    id = db.Column(db.Integer, primary_key=True)
    prolific_pid = db.Column(db.String(50), unique=True, index=True)
    study_id = db.Column(db.String(50))
    session_id = db.Column(db.String(50))

    # Demographics
    age = db.Column(db.Integer)
    gender = db.Column(db.String(50))
    education = db.Column(db.String(100))
    native_language = db.Column(db.String(50))
    country = db.Column(db.String(50))

    # Technical info
    browser = db.Column(db.String(100))
    screen_width = db.Column(db.Integer)
    screen_height = db.Column(db.Integer)

    # Survey responses
    strategy_description = db.Column(db.Text)
    suspicion_probe = db.Column(db.Text)
    additional_comments = db.Column(db.Text)

    # Relationships
    sessions = relationship(
        "ExperimentSession",
        back_populates="participant"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "prolific_pid": self.prolific_pid,
            "age": self.age,
            "gender": self.gender,
            "education": self.education,
            "native_language": self.native_language,
            "country": self.country,
        }


class ExperimentSession(db.Model, TimestampMixin):
    """
    A complete experiment session.

    Records all session-level data and aggregated metrics.
    """

    __tablename__ = "experiment_sessions"

    id = db.Column(db.Integer, primary_key=True)
    session_uuid = db.Column(db.String(36), unique=True, nullable=False, index=True)

    # Foreign keys
    participant_id = db.Column(
        db.Integer,
        db.ForeignKey("participants.id"),
        nullable=True
    )
    configuration_id = db.Column(
        db.Integer,
        db.ForeignKey("study_configurations.id"),
        nullable=False
    )

    # Experimental condition
    condition = db.Column(db.String(20), nullable=False)  # A_then_B, B_then_A, simultaneous

    # Session timing
    started_at = db.Column(db.DateTime, nullable=False)
    completed_at = db.Column(db.DateTime)
    duration_seconds = db.Column(db.Float)

    # Completion status
    completed = db.Column(db.Boolean, default=False)
    current_phase = db.Column(db.String(20), default="consent")

    # ISI condition (for Study 3b)
    isi_duration_ms = db.Column(db.Integer)

    # Training metrics
    training_completed = db.Column(db.Boolean, default=False)
    trials_to_criterion = db.Column(db.Integer)
    training_attempts = db.Column(db.Integer, default=1)

    # Manipulation check (replaces criterion check)
    manipulation_check_forward_correct = db.Column(db.Integer, default=0)
    manipulation_check_reverse_correct = db.Column(db.Integer, default=0)
    manipulation_check_passed = db.Column(db.Boolean)

    # Test metrics - Block 1 (4-AFC, CONFIRMATORY)
    forward_correct = db.Column(db.Integer, default=0)
    forward_total = db.Column(db.Integer, default=0)
    reverse_correct = db.Column(db.Integer, default=0)
    reverse_total = db.Column(db.Integer, default=0)

    # Test metrics - Block 2 (Typed Recall, EXPLORATORY)
    recall_forward_correct = db.Column(db.Integer, default=0)
    recall_forward_total = db.Column(db.Integer, default=0)
    recall_reverse_correct = db.Column(db.Integer, default=0)
    recall_reverse_total = db.Column(db.Integer, default=0)

    # Phase 5 retraining metrics (Study 3c)
    phase5_completed = db.Column(db.Boolean, default=False)
    post_retraining_forward_correct = db.Column(db.Integer, default=0)
    post_retraining_forward_total = db.Column(db.Integer, default=0)
    post_retraining_reverse_correct = db.Column(db.Integer, default=0)
    post_retraining_reverse_total = db.Column(db.Integer, default=0)

    # Attention check
    attention_check_passed = db.Column(db.Boolean)
    attention_check_response = db.Column(db.String(100))

    # Data quality flags
    excluded = db.Column(db.Boolean, default=False)
    exclusion_reason = db.Column(db.String(100))

    # Stimulus assignments (JSON mapping symbol_id -> label)
    stimulus_mapping_json = db.Column(db.Text)

    # Additional session metadata (JSON)
    metadata_json = db.Column(db.Text)

    # Relationships
    participant = relationship("Participant", back_populates="sessions")
    configuration = relationship("StudyConfiguration", back_populates="sessions")
    trials = relationship(
        "Trial",
        back_populates="session",
        order_by="Trial.trial_number"
    )

    @property
    def stimulus_mapping(self) -> Dict[str, str]:
        """Get stimulus mapping as dict."""
        if self.stimulus_mapping_json:
            return json.loads(self.stimulus_mapping_json)
        return {}

    @stimulus_mapping.setter
    def stimulus_mapping(self, value: Dict[str, str]) -> None:
        """Set stimulus mapping from dict."""
        self.stimulus_mapping_json = json.dumps(value)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get session metadata as dict."""
        if self.metadata_json:
            return json.loads(self.metadata_json)
        return {}

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        """Set session metadata from dict."""
        self.metadata_json = json.dumps(value)

    @property
    def forward_accuracy(self) -> float:
        """Calculate forward accuracy."""
        if self.forward_total == 0:
            return 0.0
        return self.forward_correct / self.forward_total

    @property
    def reverse_accuracy(self) -> float:
        """Calculate reverse accuracy."""
        if self.reverse_total == 0:
            return 0.0
        return self.reverse_correct / self.reverse_total

    @property
    def asymmetry(self) -> float:
        """Calculate asymmetry (forward - reverse)."""
        return self.forward_accuracy - self.reverse_accuracy

    @property
    def recall_forward_accuracy(self) -> float:
        """Calculate typed recall forward accuracy."""
        if self.recall_forward_total == 0:
            return 0.0
        return self.recall_forward_correct / self.recall_forward_total

    @property
    def recall_reverse_accuracy(self) -> float:
        """Calculate typed recall reverse accuracy."""
        if self.recall_reverse_total == 0:
            return 0.0
        return self.recall_reverse_correct / self.recall_reverse_total

    @property
    def recall_asymmetry(self) -> float:
        """Calculate typed recall asymmetry."""
        return self.recall_forward_accuracy - self.recall_reverse_accuracy

    def to_dict(self, include_trials: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "session_uuid": self.session_uuid,
            "condition": self.condition,
            "isi_duration_ms": self.isi_duration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "completed": self.completed,
            "current_phase": self.current_phase,
            "training_completed": self.training_completed,
            "trials_to_criterion": self.trials_to_criterion,
            "manipulation_check_passed": self.manipulation_check_passed,
            # Block 1 (4-AFC)
            "forward_correct": self.forward_correct,
            "forward_total": self.forward_total,
            "reverse_correct": self.reverse_correct,
            "reverse_total": self.reverse_total,
            "forward_accuracy": self.forward_accuracy,
            "reverse_accuracy": self.reverse_accuracy,
            "asymmetry": self.asymmetry,
            # Block 2 (Typed Recall)
            "recall_forward_correct": self.recall_forward_correct,
            "recall_forward_total": self.recall_forward_total,
            "recall_reverse_correct": self.recall_reverse_correct,
            "recall_reverse_total": self.recall_reverse_total,
            "recall_forward_accuracy": self.recall_forward_accuracy,
            "recall_reverse_accuracy": self.recall_reverse_accuracy,
            "recall_asymmetry": self.recall_asymmetry,
            # Phase 5 retraining
            "phase5_completed": self.phase5_completed,
            # Quality checks
            "attention_check_passed": self.attention_check_passed,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
        }

        if include_trials:
            result["trials"] = [t.to_dict() for t in self.trials]

        return result


class Trial(db.Model, TimestampMixin):
    """
    Individual trial data.

    Records timing, responses, and correctness for each trial.
    """

    __tablename__ = "trials"

    id = db.Column(db.Integer, primary_key=True)

    # Foreign key
    session_id = db.Column(
        db.Integer,
        db.ForeignKey("experiment_sessions.id"),
        nullable=False
    )

    # Trial identification
    trial_number = db.Column(db.Integer, nullable=False)
    phase = db.Column(db.String(30), nullable=False)
    # Phases: training, manipulation_check, test_4afc, test_recall,
    #         simultaneous_probe, phase5_retraining, phase5_test
    block = db.Column(db.Integer)

    # Stimulus
    symbol_id = db.Column(db.String(50), nullable=False)
    label = db.Column(db.String(20), nullable=False)

    # Test trial specifics
    test_direction = db.Column(db.String(20))  # forward, reverse
    test_type = db.Column(db.String(20))  # 4afc, typed_recall
    foil_labels = db.Column(db.Text)  # JSON list for 4-AFC

    # Response data
    response = db.Column(db.String(100))
    correct = db.Column(db.Boolean)
    response_time_ms = db.Column(db.Integer)

    # Confidence rating (1-4 scale)
    confidence = db.Column(db.Integer)

    # Timeout indicator
    timed_out = db.Column(db.Boolean, default=False)

    # Timing
    trial_started_at = db.Column(db.DateTime)
    trial_ended_at = db.Column(db.DateTime)

    # Presentation timing (actual vs. planned)
    fixation_actual_ms = db.Column(db.Integer)
    stimulus_actual_ms = db.Column(db.Integer)
    isi_actual_ms = db.Column(db.Integer)

    # Simultaneous probe details
    probe_type = db.Column(db.String(20))  # forward_probe, reverse_probe

    # Relationships
    session = relationship("ExperimentSession", back_populates="trials")

    @property
    def foils(self) -> List[str]:
        """Get foil labels as list."""
        if self.foil_labels:
            return json.loads(self.foil_labels)
        return []

    @foils.setter
    def foils(self, value: List[str]) -> None:
        """Set foil labels from list."""
        self.foil_labels = json.dumps(value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "trial_number": self.trial_number,
            "phase": self.phase,
            "block": self.block,
            "symbol_id": self.symbol_id,
            "label": self.label,
            "test_direction": self.test_direction,
            "test_type": self.test_type,
            "foils": self.foils,
            "response": self.response,
            "correct": self.correct,
            "response_time_ms": self.response_time_ms,
            "confidence": self.confidence,
            "timed_out": self.timed_out,
            "probe_type": self.probe_type,
            "isi_actual_ms": self.isi_actual_ms,
            "trial_started_at": (
                self.trial_started_at.isoformat()
                if self.trial_started_at else None
            ),
        }


def init_db(app):
    """Initialize database with app."""
    db.init_app(app)
    with app.app_context():
        db.create_all()


def reset_db(app):
    """Reset database (drop and recreate all tables)."""
    with app.app_context():
        db.drop_all()
        db.create_all()
