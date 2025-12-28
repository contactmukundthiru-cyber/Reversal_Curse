# Reproducibility Checklist

## For Nature Human Behaviour Submission

This document provides a comprehensive checklist to ensure full reproducibility of all analyses and figures reported in the manuscript "Temporal Credit Assignment Shapes the Direction of Knowledge Retrieval".

---

## 1. Environment Setup

### Software Requirements

✅ **Python**: 3.10 or higher (tested with 3.12.3)
✅ **Operating System**: Linux, macOS, or Windows with WSL
✅ **Memory**: Minimum 8GB RAM recommended
✅ **Disk Space**: ~5GB for data and results

### Installation

```bash
# Clone repository
git clone https://github.com/contactmukundthiru-cyber/Reversal_Curse.git
cd Reversal_Curse

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install exact dependencies
pip install -r requirements-lock.txt

# Verify installation
python -m pytest tests/unit/ -v
```

---

## 2. Data Generation

All analyses use synthetic data that simulates the expected statistical properties of real studies. To generate the complete dataset:

```bash
# Generate all synthetic datasets
python scripts/generate_synthetic_data.py
```

This creates:
- `data/raw/duolingo_learning_events.csv` - Study 1 data (~500K learning events)
- `data/raw/wikipedia_fact_pairs.csv` - Study 2 data (~5K fact pairs)
- `data/processed/experimental_data.csv` - Study 3 data (~180 participants)

### Data Integrity Checks

Run automated data validation:
```bash
python -m pytest tests/integration/test_end_to_end.py::TestDataIntegrity -v
```

---

## 3. Statistical Analyses

### Study 1: Duolingo Large-Scale Analysis

```python
from src.analysis.duolingo import DuolingoAnalyzer
from pathlib import Path

analyzer = DuolingoAnalyzer()
analyzer.load_data(Path("data/raw/duolingo_learning_events.csv"))

# Main analysis
results = analyzer.compute_reversal_gap()
print(f"Reversal Gap: {results.gap:.1%}")
print(f"Cohen's h: {results.cohens_h:.2f}")

# Export for manuscript
analyzer.export_results("data/results/study1_duolingo.json")
```

**Expected Results:**
- Forward accuracy: ~82%
- Reverse accuracy: ~41%
- Reversal gap: ~41pp
- Effect size (h): >1.0 (very large)

### Study 2: Wikipedia Factual Knowledge

```python
from src.analysis.wikipedia import WikipediaAnalyzer

analyzer = WikipediaAnalyzer()
analyzer.load_quiz_data(Path("data/raw/wikipedia_fact_pairs.csv"))

# Aggregate analysis
results = analyzer.compute_aggregate_reversal_gap()
analyzer.export_results("data/results/study2_wikipedia.json")

# By-relationship analysis
by_type = analyzer.analyze_by_relationship()
```

**Expected Results:**
- Forward accuracy: ~74%
- Reverse accuracy: ~39%
- Reversal gap: ~35pp
- Effect size (h): >0.8 (large)

### Study 3: The Flip (Main Experimental Result)

```python
from src.analysis.experimental import ExperimentalAnalyzer

analyzer = ExperimentalAnalyzer()
analyzer.load_data(Path("data/processed/experimental_data.csv"))

# Full analysis pipeline
results = analyzer.run_full_analysis()

# Generate APA-formatted summary
apa_summary = analyzer.generate_apa_summary()
print(apa_summary)

# Export
analyzer.export_results("data/results/study3_experimental.json")
```

**Expected Results:**
- **A-then-B condition**: Forward > Reverse (+47pp asymmetry)
- **B-then-A condition**: Reverse > Forward (−44pp asymmetry) **← THE FLIP**
- **Simultaneous condition**: No asymmetry (+3pp, NS)
- **ANOVA interaction**: p < .001, η²p > .25

### Power Analysis

Verify studies are adequately powered:

```python
from src.analysis.statistics import calculate_required_sample_size

# For Study 3
n_required = calculate_required_sample_size(
    expected_effect_size=0.5,  # Cohen's d
    alpha=0.05,
    power=0.80
)
print(f"Required N per condition: {n_required}")  # Should be ~64
```

---

## 4. Figure Generation

All figures follow Nature Human Behaviour formatting standards (Arial font, 300 DPI, colorblind-friendly palettes).

```python
from src.visualization.figures import FigureGenerator

generator = FigureGenerator(output_dir=Path("figures"))

# Load analysis results
import json
with open("data/results/study3_experimental.json") as f:
    exp_results = json.load(f)

# Generate all main figures
figures = generator.generate_all_figures(
    duolingo_results=None,  # Load from study 1
    wikipedia_results=None,  # Load from study 2
    experimental_results=exp_results
)

# Figures saved to figures/ directory
```

**Output Files:**
- `figure_1_empty_triangle.pdf` - Duolingo empty triangle plot
- `figure_2_domain_comparison.pdf` - Cross-domain comparison
- `figure_3_flip.pdf` - **Main result (THE FLIP)**
- `figure_4_asymmetry.pdf` - Asymmetry comparison
- `figure_5_effect_sizes.pdf` - Effect size benchmarking

---

## 5. Computational Models

### Bayesian Curse Prediction Model

```python
from src.models.bayesian import BayesianCurseModel, create_simulated_reversal

model = BayesianCurseModel(
    updating_asymmetry=0.3,
    load_sensitivity=0.5,
    tom_degradation_rate=0.4
)

# Generate simulated reversal
reversal = create_simulated_reversal(
    n_propositions=10,
    reversal_fraction=0.5
)

# Predict curse magnitude
magnitude, components = model.predict_curse_magnitude(
    reversal,
    return_components=True
)
print(f"Predicted curse: {magnitude:.2f}")
```

### Intervention Efficacy Simulation

```python
from src.models.intervention import InterventionSimulator, InterventionType

simulator = InterventionSimulator()

result = simulator.simulate_intervention(
    reversal=reversal,
    intervention_type=InterventionType.EXPLICIT_MAPPING,
    intensity=0.7,
    n_simulations=1000
)

print(f"Baseline curse: {result.baseline_curse:.2f}")
print(f"Post-intervention: {result.post_intervention_curse:.2f}")
print(f"Reduction: {result.reduction_percent:.1f}%")
```

---

## 6. Testing and Validation

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# With coverage report
pytest tests/unit/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Integration Tests

```bash
# Run end-to-end pipeline tests
pytest tests/integration/ -v -s
```

### Code Quality

```bash
# Type checking
mypy src --ignore-missing-imports

# Linting
flake8 src --max-line-length=100

# Code formatting (check only)
black src --check
```

---

## 7. Results Export and Archiving

### Export All Results

```bash
# Run complete analysis pipeline
python scripts/run_all_analyses.py

# Results saved to:
# - data/results/study1_duolingo.json
# - data/results/study2_wikipedia.json
# - data/results/study3_experimental.json
# - figures/*.pdf
```

### Archive for OSF

```bash
# Create reproducibility archive
tar -czf reversal-curse-reproducibility.tar.gz \
    data/ \
    figures/ \
    src/ \
    tests/ \
    scripts/ \
    requirements-lock.txt \
    README.md \
    docs/REPRODUCIBILITY.md \
    LICENSE

# Upload to OSF project
```

---

## 8. Computational Environment

### Hardware Specifications

**Development Environment:**
- CPU: Intel/AMD x86_64 or Apple Silicon
- RAM: 8GB minimum, 16GB recommended
- Storage: 10GB available space

**Testing Environment:**
- Ubuntu 20.04/22.04 LTS
- macOS 12+ (Monterey or later)
- Windows 10/11 with WSL2

### Exact Package Versions

See `requirements-lock.txt` for exact versions of all dependencies. Key packages:

- numpy==2.3.5
- pandas==2.3.3
- scipy==1.16.3
- statsmodels==0.14.6
- scikit-learn==1.8.0
- matplotlib==3.10.8
- seaborn==0.13.2
- pingouin==0.5.5

---

## 9. Common Issues and Solutions

### Issue: Import errors

**Solution:**
```bash
# Ensure you're in the project root
cd /path/to/reversal-curse

# Activate virtual environment
source .venv/bin/activate

# Re-install dependencies
pip install -r requirements-lock.txt
```

### Issue: Data files not found

**Solution:**
```bash
# Generate synthetic data
python scripts/generate_synthetic_data.py

# Verify data exists
ls -lh data/raw/
ls -lh data/processed/
```

### Issue: Figure generation fails

**Solution:**
```bash
# Ensure matplotlib backend is set
export MPLBACKEND=Agg

# Create figures directory
mkdir -p figures

# Re-run figure generation
python scripts/generate_figures.py
```

---

## 10. Timeline for Reproduction

**Estimated time to reproduce all analyses:**

1. Environment setup: ~10 minutes
2. Data generation: ~2 minutes
3. Statistical analyses: ~5 minutes
4. Figure generation: ~3 minutes
5. Model simulations: ~5 minutes

**Total: ~25 minutes** for complete reproduction from scratch.

---

## 11. Pre-Registration and Transparency

### Pre-Registered Analyses

See `docs/pre_registration.md` for:
- Formal hypotheses (H1-H4)
- Design specifications
- Analysis plan with decision rules
- Sample size justification

### Deviations from Pre-Registration

All deviations will be clearly documented in the manuscript with justification.

---

## 12. Data and Code Availability

- **Code**: [GitHub repository](https://github.com/contactmukundthiru-cyber/Reversal_Curse) (MIT License)
- **Data**: Synthetic datasets included in repository
- **Figures**: Generated via `scripts/run_full_pipeline.py`
- **Supplementary Materials**: Complete analysis scripts in `scripts/`

---

## 13. Contact Information

For questions about reproducibility:

- **GitHub Issues**: https://github.com/contactmukundthiru-cyber/Reversal_Curse/issues

---

## 14. Reproducibility Certification

This project follows:

- ✅ Nature Research's Code & Software Submission Guidelines
- ✅ FAIR Principles (Findable, Accessible, Interoperable, Reusable)
- ✅ TOP Guidelines (Transparency and Openness Promotion)

**Version Control:**
- All code is version-controlled with Git
- Tagged releases for each manuscript version
- Complete commit history available

**Documentation:**
- Comprehensive README
- Inline code documentation
- Pre-registration document
- This reproducibility guide

---

## 15. Citation

If you use this code or data, please cite:

```bibtex
@misc{thiru2025temporal,
  title={Temporal Credit Assignment Shapes the Direction of Knowledge Retrieval},
  author={Thiru, Mukund},
  year={2025},
  howpublished={\url{https://github.com/contactmukundthiru-cyber/Reversal_Curse}}
}
```

---

**Last Updated**: 2025-12-28

**Reproducibility Verified**: All analyses can be reproduced from scratch in <30 minutes.
