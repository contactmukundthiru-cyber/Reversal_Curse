"""
Extended API endpoints for enhanced study types.

This module provides API support for:
1. Longitudinal studies with multiple sessions
2. Intervention studies with cognitive scaffolding
3. Domain-specific studies (medical, climate)
4. Neural correlate studies with EEG/fMRI integration
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request, current_app

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.models import db, Participant, ExperimentSession, Trial
from src.models.intervention import (
    CognitiveScaffold,
    InterventionConfig,
    InterventionSimulator,
    InterventionType,
)
from src.analysis.longitudinal import (
    LongitudinalAnalyzer,
    KnowledgeSnapshot,
    ReversalTimeline,
)

logger = logging.getLogger(__name__)

# Create blueprint for extended API
api_ext = Blueprint('api_ext', __name__, url_prefix='/api/v2')


# ============================================================================
# Longitudinal Study Endpoints
# ============================================================================

@api_ext.route('/longitudinal/start', methods=['POST'])
def start_longitudinal_study():
    """
    Start a longitudinal study session.

    Longitudinal studies track participants across multiple sessions
    over days/weeks/months to measure curse decay over time.
    """
    data = request.json or {}

    participant_id = data.get('participant_id')
    domain = data.get('domain', 'general')
    session_number = data.get('session_number', 1)
    days_since_reversal = data.get('days_since_reversal')

    # Check if this is a returning participant
    participant = None
    if participant_id:
        participant = Participant.query.filter_by(prolific_pid=participant_id).first()

    if not participant:
        participant = Participant(
            prolific_pid=participant_id or str(uuid.uuid4()),
            metadata={'study_type': 'longitudinal', 'domain': domain}
        )
        db.session.add(participant)

    # Create session
    session_uuid = str(uuid.uuid4())
    exp_session = ExperimentSession(
        session_uuid=session_uuid,
        participant=participant,
        condition=f"longitudinal_{domain}",
        current_phase="instructions",
        started_at=datetime.utcnow(),
        metadata={
            'study_type': 'longitudinal',
            'domain': domain,
            'session_number': session_number,
            'days_since_reversal': days_since_reversal
        }
    )
    db.session.add(exp_session)
    db.session.commit()

    # Determine follow-up schedule
    follow_up_intervals = [1, 7, 14, 30, 90]  # Days
    next_session_days = None
    for interval in follow_up_intervals:
        if session_number <= follow_up_intervals.index(interval) + 1:
            next_session_days = interval
            break

    return jsonify({
        'success': True,
        'session_uuid': session_uuid,
        'participant_id': participant.prolific_pid,
        'session_number': session_number,
        'domain': domain,
        'follow_up_schedule': {
            'intervals_days': follow_up_intervals,
            'current_session': session_number,
            'next_session_in_days': next_session_days
        }
    })


@api_ext.route('/longitudinal/<session_uuid>/record-snapshot', methods=['POST'])
def record_knowledge_snapshot(session_uuid: str):
    """
    Record a knowledge snapshot for longitudinal tracking.

    Captures current knowledge state including:
    - Test performance (forward/reverse)
    - Confidence ratings
    - Response times
    """
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({'error': 'Session not found'}), 404

    data = request.json or {}

    snapshot_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'forward_accuracy': data.get('forward_accuracy', 0),
        'reverse_accuracy': data.get('reverse_accuracy', 0),
        'asymmetry_score': data.get('forward_accuracy', 0) - data.get('reverse_accuracy', 0),
        'confidence_ratings': data.get('confidence_ratings', {}),
        'mean_rt_forward_ms': data.get('mean_rt_forward_ms'),
        'mean_rt_reverse_ms': data.get('mean_rt_reverse_ms'),
        'session_number': exp_session.metadata.get('session_number', 1) if exp_session.metadata else 1,
        'days_since_reversal': exp_session.metadata.get('days_since_reversal') if exp_session.metadata else None
    }

    # Store in session metadata
    if not exp_session.metadata:
        exp_session.metadata = {}
    if 'snapshots' not in exp_session.metadata:
        exp_session.metadata['snapshots'] = []
    exp_session.metadata['snapshots'].append(snapshot_data)

    db.session.commit()

    return jsonify({
        'success': True,
        'snapshot': snapshot_data
    })


@api_ext.route('/longitudinal/participant/<participant_id>/trajectory', methods=['GET'])
def get_participant_trajectory(participant_id: str):
    """
    Get a participant's longitudinal trajectory.

    Returns all sessions and snapshots for trajectory analysis.
    """
    participant = Participant.query.filter_by(prolific_pid=participant_id).first()

    if not participant:
        return jsonify({'error': 'Participant not found'}), 404

    sessions = ExperimentSession.query.filter_by(participant_id=participant.id).all()

    trajectory = []
    for session in sessions:
        if session.metadata and session.metadata.get('study_type') == 'longitudinal':
            session_data = {
                'session_uuid': session.session_uuid,
                'session_number': session.metadata.get('session_number', 1),
                'days_since_reversal': session.metadata.get('days_since_reversal'),
                'completed': session.completed,
                'forward_accuracy': session.forward_accuracy,
                'reverse_accuracy': session.reverse_accuracy,
                'asymmetry': session.forward_accuracy - session.reverse_accuracy if session.forward_accuracy else None,
                'snapshots': session.metadata.get('snapshots', [])
            }
            trajectory.append(session_data)

    # Sort by session number
    trajectory.sort(key=lambda x: x.get('session_number', 0))

    return jsonify({
        'success': True,
        'participant_id': participant_id,
        'n_sessions': len(trajectory),
        'trajectory': trajectory
    })


# ============================================================================
# Intervention Study Endpoints
# ============================================================================

@api_ext.route('/intervention/start', methods=['POST'])
def start_intervention_session():
    """
    Start an intervention study session.

    Intervention studies include cognitive scaffolding to reduce
    the reversal curse effect.
    """
    data = request.json or {}

    intervention_type = data.get('intervention_type', 'explicit_mapping')
    intensity = data.get('intensity', 0.7)
    is_control = data.get('is_control', False)

    # Create session
    session_uuid = str(uuid.uuid4())
    condition = 'control' if is_control else f'intervention_{intervention_type}'

    exp_session = ExperimentSession(
        session_uuid=session_uuid,
        condition=condition,
        current_phase="instructions",
        started_at=datetime.utcnow(),
        metadata={
            'study_type': 'intervention',
            'intervention_type': intervention_type if not is_control else None,
            'intensity': intensity if not is_control else 0,
            'is_control': is_control
        }
    )
    db.session.add(exp_session)
    db.session.commit()

    # Initialize scaffold if intervention condition
    scaffold_config = None
    if not is_control:
        config = InterventionConfig(
            intensity=intensity,
            n_mapping_prompts=max(3, int(intensity * 8)),
            include_reflection=intensity > 0.5
        )
        scaffold_config = {
            'n_prompts': config.n_mapping_prompts,
            'duration_minutes': config.duration_minutes,
            'include_reflection': config.include_reflection
        }

    return jsonify({
        'success': True,
        'session_uuid': session_uuid,
        'condition': condition,
        'is_intervention': not is_control,
        'scaffold_config': scaffold_config
    })


@api_ext.route('/intervention/<session_uuid>/scaffold/prompts', methods=['GET'])
def get_scaffold_prompts(session_uuid: str):
    """
    Get cognitive scaffold mapping prompts for a session.
    """
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({'error': 'Session not found'}), 404

    if not exp_session.metadata or exp_session.metadata.get('is_control', True):
        return jsonify({'error': 'Not an intervention session'}), 400

    intensity = exp_session.metadata.get('intensity', 0.7)
    n_prompts = max(3, int(intensity * 8))

    # Generate prompts based on stimulus set
    stimulus_mapping = exp_session.stimulus_mapping or {}
    prompts = []

    prompt_templates = [
        "You learned that {symbol} means '{label}'. Explain this in your own words.",
        "If someone new was learning about {symbol}, how would you explain that it means '{label}'?",
        "Why might someone incorrectly guess what {symbol} means? The correct answer is '{label}'.",
        "Compare {symbol} to other symbols you learned. Why does it specifically mean '{label}'?",
        "Imagine teaching {symbol} = '{label}' to a friend. What would you say?"
    ]

    for i, (sym_id, mapping) in enumerate(list(stimulus_mapping.items())[:n_prompts]):
        template = prompt_templates[i % len(prompt_templates)]
        label = mapping.get('label', 'unknown') if isinstance(mapping, dict) else str(mapping)

        prompts.append({
            'id': i,
            'symbol_id': sym_id,
            'prompt': template.format(symbol=f"Symbol #{i+1}", label=label),
            'requires_explanation': True,
            'min_words': 10
        })

    return jsonify({
        'success': True,
        'prompts': prompts,
        'n_required': n_prompts
    })


@api_ext.route('/intervention/<session_uuid>/scaffold/response', methods=['POST'])
def record_scaffold_response(session_uuid: str):
    """
    Record a response to a scaffold prompt.
    """
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({'error': 'Session not found'}), 404

    data = request.json or {}

    response_data = {
        'prompt_id': data.get('prompt_id'),
        'response': data.get('response', ''),
        'response_time_seconds': data.get('response_time_seconds'),
        'word_count': len(data.get('response', '').split()),
        'timestamp': datetime.utcnow().isoformat()
    }

    # Assess response quality
    response_text = data.get('response', '').lower()
    quality_indicators = {
        'has_explanation': len(response_text.split()) > 10,
        'mentions_comparison': any(w in response_text for w in ['like', 'similar', 'different', 'compare']),
        'mentions_perspective': any(w in response_text for w in ['someone', 'person', 'they', 'friend', 'learner'])
    }
    response_data['quality_indicators'] = quality_indicators
    response_data['quality_score'] = sum(quality_indicators.values()) / len(quality_indicators)

    # Store in session
    if not exp_session.metadata:
        exp_session.metadata = {}
    if 'scaffold_responses' not in exp_session.metadata:
        exp_session.metadata['scaffold_responses'] = []
    exp_session.metadata['scaffold_responses'].append(response_data)

    db.session.commit()

    # Generate feedback
    if response_data['quality_score'] > 0.7:
        feedback = "Great explanation! Your perspective-taking will help you teach this to others."
    elif response_data['quality_score'] > 0.4:
        feedback = "Good start. Try to think about how someone new to this would understand it."
    else:
        feedback = "Try to elaborate more on why this association makes sense."

    return jsonify({
        'success': True,
        'quality_score': response_data['quality_score'],
        'feedback': feedback
    })


@api_ext.route('/intervention/<session_uuid>/predict-effect', methods=['GET'])
def predict_intervention_effect(session_uuid: str):
    """
    Predict the expected effect of the intervention based on responses.
    """
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({'error': 'Session not found'}), 404

    if not exp_session.metadata:
        return jsonify({'error': 'No intervention data'}), 400

    responses = exp_session.metadata.get('scaffold_responses', [])

    if not responses:
        return jsonify({
            'success': True,
            'predicted_reduction': 0,
            'message': 'No scaffold responses recorded yet'
        })

    # Compute engagement metrics
    avg_quality = sum(r.get('quality_score', 0) for r in responses) / len(responses)
    completion_rate = len(responses) / exp_session.metadata.get('n_prompts', 5)

    # Estimate effect based on engagement
    intensity = exp_session.metadata.get('intensity', 0.7)
    predicted_reduction = avg_quality * completion_rate * intensity * 0.4  # Max 40% reduction

    return jsonify({
        'success': True,
        'n_responses': len(responses),
        'avg_quality': avg_quality,
        'completion_rate': completion_rate,
        'predicted_reduction': predicted_reduction,
        'interpretation': f"Expected to reduce reversal curse by approximately {predicted_reduction*100:.1f}%"
    })


# ============================================================================
# Domain-Specific Study Endpoints
# ============================================================================

@api_ext.route('/domain/medical/start', methods=['POST'])
def start_medical_study():
    """
    Start a medical domain study session.

    Studies how medical guideline changes affect patient communication.
    """
    data = request.json or {}

    guideline_id = data.get('guideline_id', 'HRT_2002')
    role = data.get('role', 'physician')  # 'physician' or 'patient'
    experience_years = data.get('experience_years')

    session_uuid = str(uuid.uuid4())

    exp_session = ExperimentSession(
        session_uuid=session_uuid,
        condition=f'medical_{role}_{guideline_id}',
        current_phase='instructions',
        started_at=datetime.utcnow(),
        metadata={
            'study_type': 'domain_medical',
            'guideline_id': guideline_id,
            'role': role,
            'experience_years': experience_years
        }
    )
    db.session.add(exp_session)
    db.session.commit()

    # Get guideline information
    guideline_info = _get_medical_guideline_info(guideline_id)

    return jsonify({
        'success': True,
        'session_uuid': session_uuid,
        'role': role,
        'guideline': guideline_info,
        'study_phases': [
            'instructions',
            'baseline_knowledge',
            'reversal_presentation',
            'communication_task',
            'outcome_assessment',
            'survey',
            'complete'
        ]
    })


def _get_medical_guideline_info(guideline_id: str) -> Dict[str, Any]:
    """Get information about a medical guideline change."""
    guidelines = {
        'HRT_2002': {
            'condition': 'Menopause',
            'treatment': 'Hormone Replacement Therapy',
            'previous': 'Recommended for all menopausal women',
            'new': 'Not recommended for disease prevention, only short-term for severe symptoms',
            'reason': 'WHI trial showed increased cardiovascular and breast cancer risk'
        },
        'ASPIRIN_2019': {
            'condition': 'Cardiovascular Prevention',
            'treatment': 'Daily Low-Dose Aspirin',
            'previous': 'Recommended for adults 50-70 at CVD risk',
            'new': 'Not routinely recommended for primary prevention',
            'reason': 'Multiple trials showed bleeding risk outweighs benefit for most'
        },
        'PSA_2012': {
            'condition': 'Prostate Cancer Screening',
            'treatment': 'PSA Testing',
            'previous': 'Annual PSA screening for men over 50',
            'new': 'Do not routinely screen, shared decision-making for 55-69',
            'reason': 'Limited mortality benefit, significant harms from overdiagnosis'
        }
    }
    return guidelines.get(guideline_id, {})


@api_ext.route('/domain/climate/start', methods=['POST'])
def start_climate_study():
    """
    Start a climate domain study session.

    Studies how climate projection updates affect public understanding.
    """
    data = request.json or {}

    update_id = data.get('update_id', 'SEA_LEVEL_2021')
    audience_type = data.get('audience_type', 'general_public')

    session_uuid = str(uuid.uuid4())

    exp_session = ExperimentSession(
        session_uuid=session_uuid,
        condition=f'climate_{audience_type}_{update_id}',
        current_phase='instructions',
        started_at=datetime.utcnow(),
        metadata={
            'study_type': 'domain_climate',
            'update_id': update_id,
            'audience_type': audience_type
        }
    )
    db.session.add(exp_session)
    db.session.commit()

    # Get projection update information
    update_info = _get_climate_update_info(update_id)

    return jsonify({
        'success': True,
        'session_uuid': session_uuid,
        'audience_type': audience_type,
        'projection_update': update_info,
        'study_phases': [
            'instructions',
            'baseline_knowledge',
            'update_presentation',
            'comprehension_test',
            'attitude_assessment',
            'complete'
        ]
    })


def _get_climate_update_info(update_id: str) -> Dict[str, Any]:
    """Get information about a climate projection update."""
    updates = {
        'SEA_LEVEL_2021': {
            'phenomenon': 'Global mean sea level rise by 2100',
            'previous_estimate': '0.63 meters (AR5)',
            'new_estimate': '0.77 meters (AR6)',
            'change_direction': 'acceleration',
            'reason': 'Improved ice sheet models and observations'
        },
        'ARCTIC_ICE_2020': {
            'phenomenon': 'Arctic summer ice-free date',
            'previous_estimate': '2050',
            'new_estimate': '2035',
            'change_direction': 'acceleration',
            'reason': 'Faster-than-projected ice loss observed'
        },
        'CARBON_BUDGET_2018': {
            'phenomenon': 'Remaining carbon budget for 1.5°C',
            'previous_estimate': '400 GtCO2',
            'new_estimate': '580 GtCO2',
            'change_direction': 'expansion',
            'reason': 'Updated climate sensitivity estimates'
        }
    }
    return updates.get(update_id, {})


# ============================================================================
# Neural Study Endpoints
# ============================================================================

@api_ext.route('/neural/start', methods=['POST'])
def start_neural_study():
    """
    Start a neural correlates study session.

    These sessions are designed for EEG or fMRI data collection.
    """
    data = request.json or {}

    modality = data.get('modality', 'eeg')  # 'eeg' or 'fmri'
    run_number = data.get('run_number', 1)
    participant_id = data.get('participant_id')

    session_uuid = str(uuid.uuid4())

    exp_session = ExperimentSession(
        session_uuid=session_uuid,
        condition=f'neural_{modality}',
        current_phase='instructions',
        started_at=datetime.utcnow(),
        metadata={
            'study_type': 'neural',
            'modality': modality,
            'run_number': run_number,
            'participant_id': participant_id,
            'neural_events': []
        }
    )
    db.session.add(exp_session)
    db.session.commit()

    # Generate timing for neural study
    timing_config = _get_neural_timing(modality)

    return jsonify({
        'success': True,
        'session_uuid': session_uuid,
        'modality': modality,
        'run_number': run_number,
        'timing': timing_config
    })


def _get_neural_timing(modality: str) -> Dict[str, Any]:
    """Get timing configuration for neural studies."""
    if modality == 'fmri':
        return {
            'tr_ms': 2000,
            'stimulus_duration_ms': 2000,
            'isi_min_ms': 2000,
            'isi_max_ms': 6000,
            'fixation_duration_ms': 500,
            'response_window_ms': 3000
        }
    else:  # EEG
        return {
            'stimulus_duration_ms': 1000,
            'isi_min_ms': 800,
            'isi_max_ms': 1200,
            'baseline_duration_ms': 200,
            'epoch_duration_ms': 1000
        }


@api_ext.route('/neural/<session_uuid>/event', methods=['POST'])
def record_neural_event(session_uuid: str):
    """
    Record a neural event with precise timing.

    Events are timestamped for later alignment with neural data.
    """
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({'error': 'Session not found'}), 404

    data = request.json or {}

    event = {
        'event_type': data.get('event_type'),  # 'stimulus_onset', 'response', 'feedback'
        'timestamp_ms': data.get('timestamp_ms'),
        'trial_number': data.get('trial_number'),
        'condition': data.get('condition'),
        'stimulus_id': data.get('stimulus_id'),
        'response': data.get('response'),
        'correct': data.get('correct'),
        'rt_ms': data.get('rt_ms'),
        'trigger_code': data.get('trigger_code')  # For EEG trigger synchronization
    }

    if not exp_session.metadata:
        exp_session.metadata = {}
    if 'neural_events' not in exp_session.metadata:
        exp_session.metadata['neural_events'] = []
    exp_session.metadata['neural_events'].append(event)

    db.session.commit()

    return jsonify({
        'success': True,
        'event_id': len(exp_session.metadata['neural_events'])
    })


@api_ext.route('/neural/<session_uuid>/events', methods=['GET'])
def get_neural_events(session_uuid: str):
    """
    Get all neural events for a session (for synchronization with neural data).
    """
    exp_session = ExperimentSession.query.filter_by(session_uuid=session_uuid).first()

    if not exp_session:
        return jsonify({'error': 'Session not found'}), 404

    events = exp_session.metadata.get('neural_events', []) if exp_session.metadata else []

    return jsonify({
        'success': True,
        'session_uuid': session_uuid,
        'modality': exp_session.metadata.get('modality') if exp_session.metadata else None,
        'n_events': len(events),
        'events': events
    })


# ============================================================================
# Analytics Endpoints
# ============================================================================

@api_ext.route('/analytics/study-types', methods=['GET'])
def get_study_type_stats():
    """
    Get statistics across different study types.
    """
    sessions = ExperimentSession.query.filter_by(completed=True).all()

    stats = {
        'standard': {'count': 0, 'sessions': []},
        'longitudinal': {'count': 0, 'sessions': []},
        'intervention': {'count': 0, 'sessions': []},
        'domain_medical': {'count': 0, 'sessions': []},
        'domain_climate': {'count': 0, 'sessions': []},
        'neural': {'count': 0, 'sessions': []}
    }

    for session in sessions:
        study_type = 'standard'
        if session.metadata:
            study_type = session.metadata.get('study_type', 'standard')

        if study_type in stats:
            stats[study_type]['count'] += 1
            stats[study_type]['sessions'].append({
                'uuid': session.session_uuid,
                'condition': session.condition,
                'forward_acc': session.forward_accuracy,
                'reverse_acc': session.reverse_accuracy,
                'asymmetry': session.forward_accuracy - session.reverse_accuracy if session.forward_accuracy else None
            })

    # Compute summary stats for each type
    for study_type, data in stats.items():
        if data['count'] > 0:
            asymmetries = [s['asymmetry'] for s in data['sessions'] if s['asymmetry'] is not None]
            if asymmetries:
                import numpy as np
                data['mean_asymmetry'] = float(np.mean(asymmetries))
                data['std_asymmetry'] = float(np.std(asymmetries))

    return jsonify({
        'success': True,
        'total_sessions': len(sessions),
        'by_study_type': stats
    })


def register_extensions(app):
    """Register the extended API blueprint with the Flask app."""
    app.register_blueprint(api_ext)
    logger.info("Registered extended API endpoints at /api/v2")
