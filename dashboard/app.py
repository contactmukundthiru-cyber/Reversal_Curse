"""
Control Dashboard for the Reversal Curse Research Project.

This dashboard provides:
- Real-time experiment monitoring
- Study configuration management
- Data export and analysis
- Results visualization
- Prolific integration controls
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, flash, send_file, session
)
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_config
from src.data.models import (
    db, init_db, Participant, Trial, ExperimentSession, StudyConfiguration
)
from src.analysis.experimental import ExperimentalAnalyzer
from src.visualization.figures import FigureGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
config = get_config()

app.config["SECRET_KEY"] = config.secret_key
app.config["SQLALCHEMY_DATABASE_URI"] = config.database.sqlite_uri
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize extensions
init_db(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ============================================================================
# User Authentication (Simple for dashboard access)
# ============================================================================

class DashboardUser(UserMixin):
    """Simple user class for dashboard authentication."""

    def __init__(self, user_id: str):
        self.id = user_id


# Dashboard users - password loaded from secure configuration
# In production, DASHBOARD_PASSWORD must be set in environment
DASHBOARD_USERS = {
    "admin": generate_password_hash(config.dashboard_password)
}


@login_manager.user_loader
def load_user(user_id: str) -> Optional[DashboardUser]:
    if user_id in DASHBOARD_USERS:
        return DashboardUser(user_id)
    return None


# ============================================================================
# Authentication Routes
# ============================================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username in DASHBOARD_USERS and check_password_hash(DASHBOARD_USERS[username], password):
            user = DashboardUser(username)
            login_user(user)
            return redirect(url_for("index"))
        flash("Invalid credentials", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    """Logout."""
    logout_user()
    return redirect(url_for("login"))


# ============================================================================
# Dashboard Routes
# ============================================================================

@app.route("/")
@login_required
def index():
    """Main dashboard page."""
    # Get summary statistics
    with app.app_context():
        total_sessions = ExperimentSession.query.count()
        completed_sessions = ExperimentSession.query.filter_by(completed=True).count()
        active_configs = StudyConfiguration.query.filter_by(is_active=True).count()

        # Get recent sessions
        recent_sessions = (
            ExperimentSession.query
            .order_by(ExperimentSession.created_at.desc())
            .limit(10)
            .all()
        )

        # Calculate completion rate by condition
        condition_stats = {}
        for condition in ["A_then_B", "B_then_A", "simultaneous"]:
            sessions = ExperimentSession.query.filter_by(
                condition=condition, completed=True
            ).all()

            if sessions:
                forward_acc = np.mean([s.forward_accuracy for s in sessions])
                reverse_acc = np.mean([s.reverse_accuracy for s in sessions])
                condition_stats[condition] = {
                    "n": len(sessions),
                    "forward_accuracy": forward_acc,
                    "reverse_accuracy": reverse_acc,
                    "asymmetry": forward_acc - reverse_acc,
                }
            else:
                condition_stats[condition] = {
                    "n": 0,
                    "forward_accuracy": 0,
                    "reverse_accuracy": 0,
                    "asymmetry": 0,
                }

    return render_template(
        "dashboard.html",
        total_sessions=total_sessions,
        completed_sessions=completed_sessions,
        active_configs=active_configs,
        recent_sessions=recent_sessions,
        condition_stats=condition_stats,
    )


@app.route("/studies")
@login_required
def studies():
    """Study configurations management."""
    configs = StudyConfiguration.query.all()
    return render_template("studies.html", configs=configs)


@app.route("/studies/create", methods=["GET", "POST"])
@login_required
def create_study():
    """Create a new study configuration."""
    if request.method == "POST":
        try:
            new_config = StudyConfiguration(
                name=request.form.get("name"),
                description=request.form.get("description"),
                is_active=request.form.get("is_active") == "on",
                n_symbol_label_pairs=int(request.form.get("n_pairs", 16)),
                training_repetitions=int(request.form.get("training_reps", 6)),
                target_n_per_condition=int(request.form.get("target_n", 60)),
                payment_usd=float(request.form.get("payment", 2.50)),
            )
            db.session.add(new_config)
            db.session.commit()
            flash("Study configuration created successfully", "success")
            return redirect(url_for("studies"))
        except Exception as e:
            flash(f"Error creating configuration: {e}", "error")

    return render_template("create_study.html")


@app.route("/studies/<int:config_id>/activate", methods=["POST"])
@login_required
def activate_study(config_id: int):
    """Activate a study configuration."""
    # Deactivate all other configs
    StudyConfiguration.query.update({StudyConfiguration.is_active: False})

    # Activate this one
    config = StudyConfiguration.query.get_or_404(config_id)
    config.is_active = True
    db.session.commit()

    flash(f"Study '{config.name}' is now active", "success")
    return redirect(url_for("studies"))


@app.route("/sessions")
@login_required
def sessions():
    """View all experiment sessions."""
    page = request.args.get("page", 1, type=int)
    per_page = 50

    query = ExperimentSession.query.order_by(ExperimentSession.created_at.desc())

    # Filtering
    condition = request.args.get("condition")
    if condition:
        query = query.filter_by(condition=condition)

    completed = request.args.get("completed")
    if completed == "true":
        query = query.filter_by(completed=True)
    elif completed == "false":
        query = query.filter_by(completed=False)

    pagination = query.paginate(page=page, per_page=per_page)

    return render_template(
        "sessions.html",
        sessions=pagination.items,
        pagination=pagination,
    )


@app.route("/sessions/<int:session_id>")
@login_required
def session_detail(session_id: int):
    """View detailed session information."""
    exp_session = ExperimentSession.query.get_or_404(session_id)
    return render_template("session_detail.html", session=exp_session)


@app.route("/analysis")
@login_required
def analysis():
    """Analysis and results page."""
    # Run analysis on completed sessions
    completed_sessions = ExperimentSession.query.filter_by(completed=True).all()

    if not completed_sessions:
        return render_template("analysis.html", results=None)

    # Prepare data for analysis
    data = []
    for session in completed_sessions:
        data.append({
            "participant_id": session.session_uuid,
            "condition": session.condition,
            "forward_correct": session.forward_correct,
            "forward_total": session.forward_total,
            "reverse_correct": session.reverse_correct,
            "reverse_total": session.reverse_total,
            "completion_time": session.duration_seconds or 0,
            "attention_check_passed": session.attention_check_passed or True,
            "completed": session.completed,
        })

    df = pd.DataFrame(data)

    # Calculate summary statistics
    summary = {
        "total_participants": len(df),
        "by_condition": {},
    }

    for condition in df["condition"].unique():
        cond_df = df[df["condition"] == condition]
        forward_acc = (cond_df["forward_correct"] / cond_df["forward_total"]).mean()
        reverse_acc = (cond_df["reverse_correct"] / cond_df["reverse_total"]).mean()

        summary["by_condition"][condition] = {
            "n": len(cond_df),
            "forward_accuracy": forward_acc,
            "reverse_accuracy": reverse_acc,
            "asymmetry": forward_acc - reverse_acc,
        }

    return render_template("analysis.html", results=summary)


@app.route("/export")
@login_required
def export_page():
    """Data export page."""
    return render_template("export.html")


@app.route("/api/export/csv")
@login_required
def export_csv():
    """Export data as CSV."""
    sessions = ExperimentSession.query.filter_by(completed=True).all()

    data = []
    for session in sessions:
        data.append({
            "session_uuid": session.session_uuid,
            "condition": session.condition,
            "forward_correct": session.forward_correct,
            "forward_total": session.forward_total,
            "reverse_correct": session.reverse_correct,
            "reverse_total": session.reverse_total,
            "forward_accuracy": session.forward_accuracy,
            "reverse_accuracy": session.reverse_accuracy,
            "asymmetry": session.asymmetry,
            "duration_seconds": session.duration_seconds,
            "attention_check_passed": session.attention_check_passed,
            "completed_at": session.completed_at,
        })

    df = pd.DataFrame(data)

    # Save to temporary file
    export_path = Path("data/exports")
    export_path.mkdir(parents=True, exist_ok=True)
    filename = f"experiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = export_path / filename
    df.to_csv(filepath, index=False)

    return send_file(filepath, as_attachment=True)


@app.route("/api/export/json")
@login_required
def export_json():
    """Export data as JSON."""
    sessions = ExperimentSession.query.filter_by(completed=True).all()
    data = [s.to_dict(include_trials=True) for s in sessions]

    export_path = Path("data/exports")
    export_path.mkdir(parents=True, exist_ok=True)
    filename = f"experiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = export_path / filename

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return send_file(filepath, as_attachment=True)


@app.route("/figures")
@login_required
def figures():
    """Generate and view figures."""
    return render_template("figures.html")


@app.route("/api/generate-figures", methods=["POST"])
@login_required
def generate_figures():
    """Generate publication figures."""
    try:
        output_dir = Path("data/figures")
        generator = FigureGenerator(output_dir)

        # Get experimental results
        sessions = ExperimentSession.query.filter_by(completed=True).all()

        if sessions:
            # Prepare results dict
            condition_results = {}
            for condition in ["A_then_B", "B_then_A", "simultaneous"]:
                cond_sessions = [s for s in sessions if s.condition == condition]
                if cond_sessions:
                    from dataclasses import dataclass

                    @dataclass
                    class MockConditionResults:
                        forward_accuracy: float
                        reverse_accuracy: float
                        asymmetry: float

                    forward_acc = np.mean([s.forward_accuracy for s in cond_sessions])
                    reverse_acc = np.mean([s.reverse_accuracy for s in cond_sessions])
                    condition_results[condition] = MockConditionResults(
                        forward_accuracy=forward_acc,
                        reverse_accuracy=reverse_acc,
                        asymmetry=forward_acc - reverse_acc,
                    )

            experimental_results = {"condition_results": condition_results}
        else:
            experimental_results = None

        figures = generator.generate_all_figures(
            experimental_results=experimental_results
        )

        figure_list = []
        for name, paths in figures.items():
            figure_list.append({
                "name": name,
                "paths": [str(p) for p in paths],
            })

        return jsonify({
            "success": True,
            "figures": figure_list,
        })

    except Exception as e:
        logger.error(f"Error generating figures: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route("/api/dashboard-stats")
@login_required
def dashboard_stats():
    """Get real-time dashboard statistics."""
    # Sessions in last 24 hours
    yesterday = datetime.utcnow() - timedelta(days=1)

    recent_count = ExperimentSession.query.filter(
        ExperimentSession.created_at >= yesterday
    ).count()

    completed_count = ExperimentSession.query.filter_by(completed=True).count()

    # Per-condition stats
    condition_data = []
    for condition in ["A_then_B", "B_then_A", "simultaneous"]:
        sessions = ExperimentSession.query.filter_by(
            condition=condition, completed=True
        ).all()

        if sessions:
            condition_data.append({
                "condition": condition,
                "n": len(sessions),
                "forward_accuracy": np.mean([s.forward_accuracy for s in sessions]),
                "reverse_accuracy": np.mean([s.reverse_accuracy for s in sessions]),
            })

    return jsonify({
        "recent_sessions": recent_count,
        "completed_sessions": completed_count,
        "condition_stats": condition_data,
        "timestamp": datetime.utcnow().isoformat(),
    })


# ============================================================================
# Enhanced Study Type Routes
# ============================================================================

@app.route("/studies/longitudinal")
@login_required
def longitudinal_studies():
    """Manage longitudinal studies."""
    # Get longitudinal sessions
    longitudinal_sessions = ExperimentSession.query.filter(
        ExperimentSession.condition.like('longitudinal%')
    ).all()

    # Group by participant
    participant_trajectories = {}
    for session in longitudinal_sessions:
        pid = session.participant.prolific_pid if session.participant else 'unknown'
        if pid not in participant_trajectories:
            participant_trajectories[pid] = []
        participant_trajectories[pid].append({
            'session_uuid': session.session_uuid,
            'session_number': session.metadata.get('session_number', 1) if session.metadata else 1,
            'days_since_reversal': session.metadata.get('days_since_reversal') if session.metadata else None,
            'forward_accuracy': session.forward_accuracy,
            'reverse_accuracy': session.reverse_accuracy,
            'asymmetry': session.forward_accuracy - session.reverse_accuracy if session.forward_accuracy else None,
            'completed': session.completed
        })

    return render_template(
        "longitudinal.html",
        participant_trajectories=participant_trajectories,
        n_participants=len(participant_trajectories),
        n_sessions=len(longitudinal_sessions)
    )


@app.route("/studies/intervention")
@login_required
def intervention_studies():
    """Manage intervention studies."""
    # Get intervention sessions
    intervention_sessions = ExperimentSession.query.filter(
        ExperimentSession.condition.like('intervention%') |
        ExperimentSession.condition.like('control%')
    ).all()

    # Separate by condition type
    intervention_data = []
    control_data = []

    for session in intervention_sessions:
        if session.completed:
            session_info = {
                'session_uuid': session.session_uuid,
                'intervention_type': session.metadata.get('intervention_type') if session.metadata else None,
                'intensity': session.metadata.get('intensity', 0) if session.metadata else 0,
                'forward_accuracy': session.forward_accuracy,
                'reverse_accuracy': session.reverse_accuracy,
                'asymmetry': session.forward_accuracy - session.reverse_accuracy if session.forward_accuracy else None,
                'scaffold_responses': len(session.metadata.get('scaffold_responses', [])) if session.metadata else 0
            }

            if session.metadata and session.metadata.get('is_control'):
                control_data.append(session_info)
            else:
                intervention_data.append(session_info)

    # Compute summary
    summary = {
        'intervention': {
            'n': len(intervention_data),
            'mean_asymmetry': np.mean([d['asymmetry'] for d in intervention_data if d['asymmetry']]) if intervention_data else 0
        },
        'control': {
            'n': len(control_data),
            'mean_asymmetry': np.mean([d['asymmetry'] for d in control_data if d['asymmetry']]) if control_data else 0
        }
    }

    if intervention_data and control_data:
        summary['effect_size'] = summary['control']['mean_asymmetry'] - summary['intervention']['mean_asymmetry']
    else:
        summary['effect_size'] = None

    return render_template(
        "intervention.html",
        intervention_data=intervention_data,
        control_data=control_data,
        summary=summary
    )


@app.route("/studies/domain")
@login_required
def domain_studies():
    """Manage domain-specific studies (medical, climate)."""
    # Get domain sessions
    medical_sessions = ExperimentSession.query.filter(
        ExperimentSession.condition.like('medical%')
    ).filter_by(completed=True).all()

    climate_sessions = ExperimentSession.query.filter(
        ExperimentSession.condition.like('climate%')
    ).filter_by(completed=True).all()

    # Medical domain summary
    medical_summary = {}
    for session in medical_sessions:
        guideline = session.metadata.get('guideline_id', 'unknown') if session.metadata else 'unknown'
        if guideline not in medical_summary:
            medical_summary[guideline] = {
                'n': 0,
                'confusion_rates': [],
                'credibility_scores': []
            }
        medical_summary[guideline]['n'] += 1

    # Climate domain summary
    climate_summary = {}
    for session in climate_sessions:
        update_id = session.metadata.get('update_id', 'unknown') if session.metadata else 'unknown'
        if update_id not in climate_summary:
            climate_summary[update_id] = {
                'n': 0,
                'confusion_rates': [],
                'credibility_scores': []
            }
        climate_summary[update_id]['n'] += 1

    return render_template(
        "domain_studies.html",
        medical_sessions=len(medical_sessions),
        climate_sessions=len(climate_sessions),
        medical_summary=medical_summary,
        climate_summary=climate_summary
    )


@app.route("/studies/neural")
@login_required
def neural_studies():
    """Manage neural correlates studies."""
    neural_sessions = ExperimentSession.query.filter(
        ExperimentSession.condition.like('neural%')
    ).all()

    # Group by modality
    fmri_sessions = []
    eeg_sessions = []

    for session in neural_sessions:
        modality = session.metadata.get('modality') if session.metadata else 'unknown'
        session_info = {
            'session_uuid': session.session_uuid,
            'participant_id': session.metadata.get('participant_id') if session.metadata else None,
            'run_number': session.metadata.get('run_number', 1) if session.metadata else 1,
            'n_events': len(session.metadata.get('neural_events', [])) if session.metadata else 0,
            'completed': session.completed
        }

        if modality == 'fmri':
            fmri_sessions.append(session_info)
        else:
            eeg_sessions.append(session_info)

    return render_template(
        "neural_studies.html",
        fmri_sessions=fmri_sessions,
        eeg_sessions=eeg_sessions,
        n_fmri=len(fmri_sessions),
        n_eeg=len(eeg_sessions)
    )


@app.route("/api/study-types/summary")
@login_required
def study_types_summary():
    """Get summary statistics across all study types."""
    all_sessions = ExperimentSession.query.filter_by(completed=True).all()

    summary = {
        'standard': {'n': 0, 'mean_asymmetry': []},
        'longitudinal': {'n': 0, 'mean_asymmetry': []},
        'intervention': {'n': 0, 'mean_asymmetry': []},
        'control': {'n': 0, 'mean_asymmetry': []},
        'medical': {'n': 0, 'mean_asymmetry': []},
        'climate': {'n': 0, 'mean_asymmetry': []},
        'neural': {'n': 0, 'mean_asymmetry': []}
    }

    for session in all_sessions:
        study_type = 'standard'
        if session.metadata:
            st = session.metadata.get('study_type', 'standard')
            if st == 'intervention':
                if session.metadata.get('is_control'):
                    study_type = 'control'
                else:
                    study_type = 'intervention'
            elif st == 'domain_medical':
                study_type = 'medical'
            elif st == 'domain_climate':
                study_type = 'climate'
            else:
                study_type = st

        if study_type in summary:
            summary[study_type]['n'] += 1
            if session.forward_accuracy is not None and session.reverse_accuracy is not None:
                summary[study_type]['mean_asymmetry'].append(
                    session.forward_accuracy - session.reverse_accuracy
                )

    # Compute means
    for key in summary:
        if summary[key]['mean_asymmetry']:
            summary[key]['mean_asymmetry'] = float(np.mean(summary[key]['mean_asymmetry']))
        else:
            summary[key]['mean_asymmetry'] = None

    return jsonify({
        'success': True,
        'summary': summary,
        'total_completed': len(all_sessions)
    })


@app.route("/api/longitudinal/trajectory/<participant_id>")
@login_required
def get_trajectory_data(participant_id: str):
    """Get trajectory data for visualization."""
    participant = Participant.query.filter_by(prolific_pid=participant_id).first()

    if not participant:
        return jsonify({'error': 'Participant not found'}), 404

    sessions = ExperimentSession.query.filter_by(participant_id=participant.id).all()

    trajectory = []
    for session in sessions:
        if session.metadata and session.metadata.get('study_type') == 'longitudinal':
            trajectory.append({
                'days_since_reversal': session.metadata.get('days_since_reversal', 0),
                'asymmetry': session.forward_accuracy - session.reverse_accuracy if session.forward_accuracy else None,
                'forward_accuracy': session.forward_accuracy,
                'reverse_accuracy': session.reverse_accuracy
            })

    trajectory.sort(key=lambda x: x['days_since_reversal'] or 0)

    return jsonify({
        'success': True,
        'participant_id': participant_id,
        'trajectory': trajectory
    })


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template("error.html", error="Internal server error"), 500


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,  # Different port from experiment app
        debug=config.debug
    )
