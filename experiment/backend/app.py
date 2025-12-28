"""
Flask backend for the Reversal Curse experimental platform.

This module provides:
- Experiment API endpoints
- Session management
- Data collection and storage
- Prolific integration
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, render_template, redirect, url_for, session
from flask_cors import CORS

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import get_config
from src.data.models import (
    db,
    init_db,
    Participant,
    Trial,
    ExperimentSession,
    Stimulus,
    StudyConfiguration,
)
from src.experiment.stimuli import StimulusGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

# Configuration
config = get_config()
app.config["SECRET_KEY"] = config.secret_key
app.config["SQLALCHEMY_DATABASE_URI"] = config.database.sqlite_uri
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_COOKIE_SECURE"] = config.env == "production"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=config.session_lifetime_hours)

# Initialize extensions
# CORS: Restrict to same origin in production, allow all in development
if config.env == "production":
    # Production: only allow specific origins
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
    allowed_origins = [o.strip() for o in allowed_origins if o.strip()]
    if allowed_origins:
        CORS(app, origins=allowed_origins, supports_credentials=True)
    else:
        # No CORS if no allowed origins specified (same-origin only)
        pass
else:
    # Development: allow all origins for testing
    CORS(app)

init_db(app)

# Stimulus generator
stimulus_generator = StimulusGenerator()


# ============================================================================
# Helper Functions
# ============================================================================

def get_or_create_active_config() -> StudyConfiguration:
    """Get active study configuration or create default."""
    active_config = StudyConfiguration.query.filter_by(is_active=True).first()

    if not active_config:
        active_config = StudyConfiguration(
            name="Default Study",
            description="Default reversal curse experiment configuration",
            is_active=True,
            n_symbol_label_pairs=config.experiment.n_symbol_label_pairs,
            training_repetitions=config.experiment.training_repetitions,
            fixation_duration_ms=config.experiment.fixation_duration_ms,
            stimulus_duration_ms=config.experiment.stimulus_duration_ms,
            blank_duration_ms=config.experiment.blank_duration_ms,
            simultaneous_duration_ms=config.experiment.simultaneous_duration_ms,
            iti_duration_ms=config.experiment.iti_duration_ms,
            test_response_deadline_ms=config.experiment.test_response_deadline_ms,
            criterion_threshold=config.experiment.criterion_threshold,
            conditions=config.experiment.conditions,
            target_n_per_condition=config.experiment.target_n_per_condition,
        )
        db.session.add(active_config)
        db.session.commit()

    return active_config


def generate_stimulus_set(n_pairs: int = 16) -> Dict[str, str]:
    """Generate a random stimulus set for a participant."""
    symbols = stimulus_generator.generate_symbols(n_pairs)
    labels = stimulus_generator.generate_labels(n_pairs)

    # Random pairing
    import random
    random.shuffle(labels)

    return {
        sym["id"]: {
            "symbol": sym,
            "label": lab
        }
        for sym, lab in zip(symbols, labels)
    }


# ============================================================================
# Experiment Routes
# ============================================================================

@app.route("/")
def index():
    """Landing page / consent form."""
    return render_template("index.html")


@app.route("/api/start-session", methods=["POST"])
def start_session():
    """
    Start a new experiment session.

    Expects Prolific parameters (PROLIFIC_PID, STUDY_ID, SESSION_ID).
    """
    data = request.json or {}

    # Get Prolific parameters
    prolific_pid = data.get("prolific_pid") or request.args.get("PROLIFIC_PID")
    study_id = data.get("study_id") or request.args.get("STUDY_ID")
    session_id_param = data.get("session_id") or request.args.get("SESSION_ID")

    # Get active configuration
    study_config = get_or_create_active_config()

    # Check if participant already exists
    participant = None
    if prolific_pid:
        participant = Participant.query.filter_by(prolific_pid=prolific_pid).first()
        if not participant:
            participant = Participant(
                prolific_pid=prolific_pid,
                study_id=study_id,
                session_id=session_id_param,
            )
            db.session.add(participant)

    # Allocate condition
    condition = study_config.allocate_condition()

    # Generate stimulus set
    stimulus_set = generate_stimulus_set(study_config.n_symbol_label_pairs)

    # Create session
    exp_session = ExperimentSession(
        session_uuid=str(uuid.uuid4()),
        participant=participant,
        configuration_id=study_config.id,
        condition=condition,
        started_at=datetime.utcnow(),
        current_phase="instructions",
        stimulus_mapping=stimulus_set,
    )
    db.session.add(exp_session)
    db.session.commit()

    # Store session ID in Flask session
    session["experiment_session_id"] = exp_session.id
    session.permanent = True

    logger.info(f"Started session {exp_session.session_uuid} in condition {condition}")

    return jsonify({
        "success": True,
        "session_uuid": exp_session.session_uuid,
        "condition": condition,
        "config": {
            "n_pairs": study_config.n_symbol_label_pairs,
            "training_repetitions": study_config.training_repetitions,
            "fixation_duration_ms": study_config.fixation_duration_ms,
            "stimulus_duration_ms": study_config.stimulus_duration_ms,
            "blank_duration_ms": study_config.blank_duration_ms,
            "simultaneous_duration_ms": study_config.simultaneous_duration_ms,
            "iti_duration_ms": study_config.iti_duration_ms,
            "test_response_deadline_ms": study_config.test_response_deadline_ms,
            "criterion_threshold": study_config.criterion_threshold,
        },
        "stimuli": stimulus_set,
    })


@app.route("/api/session/<session_uuid>", methods=["GET"])
def get_session(session_uuid: str):
    """Get current session state."""
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({"error": "Session not found"}), 404

    return jsonify({
        "success": True,
        "session": exp_session.to_dict(),
    })


@app.route("/api/session/<session_uuid>/update-phase", methods=["POST"])
def update_phase(session_uuid: str):
    """Update the current phase of the experiment."""
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({"error": "Session not found"}), 404

    data = request.json or {}
    new_phase = data.get("phase")

    valid_phases = [
        "consent", "instructions", "training", "criterion",
        "distractor", "test", "survey", "debrief", "complete"
    ]

    if new_phase not in valid_phases:
        return jsonify({"error": f"Invalid phase: {new_phase}"}), 400

    exp_session.current_phase = new_phase

    if new_phase == "complete":
        exp_session.completed = True
        exp_session.completed_at = datetime.utcnow()
        exp_session.duration_seconds = (
            exp_session.completed_at - exp_session.started_at
        ).total_seconds()

    db.session.commit()

    return jsonify({
        "success": True,
        "phase": new_phase,
    })


@app.route("/api/session/<session_uuid>/record-trial", methods=["POST"])
def record_trial(session_uuid: str):
    """Record a single trial."""
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({"error": "Session not found"}), 404

    data = request.json or {}

    trial = Trial(
        session_id=exp_session.id,
        trial_number=data.get("trial_number", 0),
        phase=data.get("phase", "training"),
        block=data.get("block"),
        symbol_id=data.get("symbol_id", ""),
        label=data.get("label", ""),
        test_direction=data.get("test_direction"),
        response=data.get("response"),
        correct=data.get("correct"),
        response_time_ms=data.get("response_time_ms"),
        trial_started_at=datetime.utcnow(),
    )

    if data.get("foils"):
        trial.foils = data["foils"]

    db.session.add(trial)

    # Update session aggregates for test trials
    if data.get("phase") == "test" and data.get("correct") is not None:
        if data.get("test_direction") == "forward":
            exp_session.forward_total += 1
            if data.get("correct"):
                exp_session.forward_correct += 1
        elif data.get("test_direction") == "reverse":
            exp_session.reverse_total += 1
            if data.get("correct"):
                exp_session.reverse_correct += 1

    db.session.commit()

    return jsonify({
        "success": True,
        "trial_id": trial.id,
    })


@app.route("/api/session/<session_uuid>/record-training", methods=["POST"])
def record_training(session_uuid: str):
    """Record training phase completion."""
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({"error": "Session not found"}), 404

    data = request.json or {}

    exp_session.training_completed = True
    exp_session.trials_to_criterion = data.get("trials_to_criterion")
    exp_session.criterion_met = data.get("criterion_met", False)
    exp_session.training_attempts = data.get("training_attempts", 1)

    db.session.commit()

    return jsonify({
        "success": True,
    })


@app.route("/api/session/<session_uuid>/record-attention", methods=["POST"])
def record_attention(session_uuid: str):
    """Record attention check result."""
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({"error": "Session not found"}), 404

    data = request.json or {}

    exp_session.attention_check_passed = data.get("passed", False)
    exp_session.attention_check_response = data.get("response", "")

    db.session.commit()

    return jsonify({
        "success": True,
    })


@app.route("/api/session/<session_uuid>/record-demographics", methods=["POST"])
def record_demographics(session_uuid: str):
    """Record participant demographics from survey."""
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({"error": "Session not found"}), 404

    data = request.json or {}

    if exp_session.participant:
        exp_session.participant.age = data.get("age")
        exp_session.participant.gender = data.get("gender")
        exp_session.participant.education = data.get("education")
        exp_session.participant.native_language = data.get("native_language")
        exp_session.participant.strategy_description = data.get("strategy")
        exp_session.participant.suspicion_probe = data.get("suspicion")
        exp_session.participant.additional_comments = data.get("comments")

    db.session.commit()

    return jsonify({
        "success": True,
    })


@app.route("/api/session/<session_uuid>/complete", methods=["POST"])
def complete_session(session_uuid: str):
    """Mark session as complete and generate completion code."""
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({"error": "Session not found"}), 404

    exp_session.completed = True
    exp_session.completed_at = datetime.utcnow()
    exp_session.current_phase = "complete"

    if exp_session.started_at:
        exp_session.duration_seconds = (
            exp_session.completed_at - exp_session.started_at
        ).total_seconds()

    db.session.commit()

    # Generate completion code for Prolific
    completion_code = f"RC-{exp_session.id:06d}-{uuid.uuid4().hex[:8].upper()}"

    return jsonify({
        "success": True,
        "completion_code": completion_code,
        "duration_minutes": exp_session.duration_seconds / 60 if exp_session.duration_seconds else None,
    })


# ============================================================================
# Data Export Routes (Protected)
# ============================================================================

def require_api_key(f):
    """Decorator to require API key for protected endpoints."""
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
        expected_key = os.getenv("EXPORT_API_KEY")

        if not expected_key:
            # If no API key is configured, only allow in development
            if config.env == "production":
                logger.warning("EXPORT_API_KEY not configured in production")
                return jsonify({"error": "Export API not configured"}), 503
            else:
                # Development mode - allow without key
                logger.debug("Export API accessed without key (development mode)")
                return f(*args, **kwargs)

        if api_key != expected_key:
            logger.warning(f"Invalid API key attempt from {request.remote_addr}")
            return jsonify({"error": "Invalid or missing API key"}), 401

        return f(*args, **kwargs)

    return decorated


@app.route("/api/export/sessions", methods=["GET"])
@require_api_key
def export_sessions():
    """Export all completed sessions.

    Requires X-API-Key header or api_key query parameter in production.
    """
    sessions = ExperimentSession.query.filter_by(completed=True).all()

    return jsonify({
        "success": True,
        "count": len(sessions),
        "sessions": [s.to_dict(include_trials=True) for s in sessions],
    })


@app.route("/api/export/summary", methods=["GET"])
@require_api_key
def export_summary():
    """Export summary statistics.

    Requires X-API-Key header or api_key query parameter in production.
    """
    sessions = ExperimentSession.query.filter_by(completed=True).all()

    by_condition = {}
    for session in sessions:
        cond = session.condition
        if cond not in by_condition:
            by_condition[cond] = {
                "count": 0,
                "forward_acc_sum": 0,
                "reverse_acc_sum": 0,
            }
        by_condition[cond]["count"] += 1
        by_condition[cond]["forward_acc_sum"] += session.forward_accuracy or 0
        by_condition[cond]["reverse_acc_sum"] += session.reverse_accuracy or 0

    summary = {}
    for cond, data in by_condition.items():
        n = data["count"]
        summary[cond] = {
            "n": n,
            "forward_accuracy": data["forward_acc_sum"] / n if n > 0 else 0,
            "reverse_accuracy": data["reverse_acc_sum"] / n if n > 0 else 0,
        }
        summary[cond]["asymmetry"] = (
            summary[cond]["forward_accuracy"] - summary[cond]["reverse_accuracy"]
        )

    return jsonify({
        "success": True,
        "total_completed": len(sessions),
        "by_condition": summary,
    })


# ============================================================================
# Experiment Page Routes
# ============================================================================

@app.route("/experiment")
def experiment():
    """Main experiment page."""
    return render_template("experiment.html")


@app.route("/complete")
def complete():
    """Completion page."""
    return render_template("complete.html")


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# Register Extended API
# ============================================================================

try:
    from experiment.backend.api_extensions import register_extensions
    register_extensions(app)
except ImportError:
    logger.warning("Extended API not available - running in basic mode")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug
    )
