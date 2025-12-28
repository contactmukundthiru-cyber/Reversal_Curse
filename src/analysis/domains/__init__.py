"""
Real-world domain modules for reversal curse research.

This package provides specialized analysis frameworks for high-stakes
domains where the reversal curse has significant practical implications:

1. Medical Diagnosis: How treatment guideline changes affect patient care
2. Climate Science: How updated projections affect public communication
3. Legal/Policy: How precedent reversals affect case outcomes
4. Educational: How curriculum changes affect teaching effectiveness
"""

from .medical import (
    MedicalReversalAnalyzer,
    TreatmentGuidelineChange,
    PatientCommunicationStudy,
)

from .climate import (
    ClimateReversalAnalyzer,
    ProjectionUpdate,
    PublicCommunicationStudy,
)

__all__ = [
    "MedicalReversalAnalyzer",
    "TreatmentGuidelineChange",
    "PatientCommunicationStudy",
    "ClimateReversalAnalyzer",
    "ProjectionUpdate",
    "PublicCommunicationStudy",
]
