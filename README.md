# Temporal Credit Assignment in Human Associative Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Pre-Data Collection](https://img.shields.io/badge/Status-Pre--Data%20Collection-yellow.svg)]()

**A research framework investigating how temporal credit assignment shapes the direction of knowledge retrieval.**

## Project Status

> **IMPORTANT**: This project is in the **pre-data collection phase**. The experimental platform is functional, but **no real participant data has been collected yet**. All analyses require real data from actual experimental participants.

### What's Ready
- Experimental platform (Flask-based web experiment)
- Pre-registration document with hypotheses and analysis plan
- Statistical analysis pipelines
- Stimulus generation (SVG symbols, pronounceable nonwords)
- Data collection infrastructure

### What's Needed Before Running the Experiment
1. **IRB/Ethics Approval** - Required for human subjects research
2. **Participant Recruitment** - Via Prolific or similar platform
3. **Pilot Testing** - Verify timing and instructions work correctly

## The Core Hypothesis

**Temporal Credit Assignment (TCA)**: Human associative learning assigns credit backward in time. When outcome B follows cue A, the learning system strengthens the A→B mapping because A *predicted* B. The reverse mapping (B→A) is not strengthened.

### The Critical Test: "The Flip"

If temporal ordering determines inference direction:

```
A-then-B training → Forward > Reverse accuracy
B-then-A training → Reverse > Forward accuracy (THE FLIP!)
Simultaneous     → Forward ≈ Reverse (no asymmetry)
```

This is a **causal manipulation** that reverses the direction of asymmetry while holding stimuli and test demands constant.

## Experimental Design

### Study 3: Controlled Experiment

- **Design**: 2 × 3 Mixed Factorial
  - Within-subjects: Test Direction (Forward vs. Reverse)
  - Between-subjects: Training Condition (A-then-B vs. B-then-A vs. Simultaneous)
- **N**: 180 participants (60 per condition)
- **Duration**: ~12 minutes per participant
- **Test Format**: Matched 4-AFC for both directions (eliminates recall/recognition confound)

### Methodological Features

1. **Matched Test Formats**: Both forward and reverse tests use 4-AFC selection
2. **Confidence Ratings**: Collected after each test trial
3. **Procedural Stimuli**: Unique symbol-label pairs generated per participant
4. **Attention Checks**: Embedded to ensure data quality

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/reversal-curse.git
cd reversal-curse

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Experiment Locally

```bash
# Start the experiment server
cd experiment/backend
python app.py

# Access at http://localhost:5000
```

### Testing the Experiment

You can test the full experiment flow by:
1. Opening http://localhost:5000 in your browser
2. Consenting and going through training
3. Completing the test phase
4. Submitting the survey

Data will be stored in `data/reversal_curse.db` (SQLite).

## Project Structure

```
reversal_curse/
├── config/                      # Configuration
│   └── settings.py             # Experiment parameters
│
├── src/                         # Core code
│   ├── analysis/               # Statistical analysis
│   │   ├── statistics.py      # Core statistics
│   │   └── experimental.py    # Experiment analysis
│   ├── data/                   # Data models
│   │   └── models.py          # SQLAlchemy models
│   └── experiment/             # Experiment components
│       └── stimuli.py         # Stimulus generation
│
├── experiment/                  # Web experiment
│   ├── frontend/              # HTML/CSS/JS
│   └── backend/               # Flask server
│
├── data/                        # Data storage (empty until data collection)
│   ├── raw/
│   ├── processed/
│   └── results/
│
├── docs/                        # Documentation
│   └── pre_registration.md    # Pre-registration document
│
└── tests/                       # Test suite
```

## Pre-Registration

See [docs/pre_registration.md](docs/pre_registration.md) for:
- Formal hypotheses (H1-H8)
- Design specifications
- Analysis plan with decision rules
- Exclusion criteria
- Sample size justification

## Before Collecting Data

### 1. Obtain IRB Approval

Human subjects research requires ethics board approval. Prepare:
- Study protocol
- Consent form (template in experiment/frontend/templates/index.html)
- Risk assessment
- Data management plan

### 2. Set Up Prolific (or alternative)

Configure for participant recruitment:
```env
PROLIFIC_API_TOKEN=your-token
```

### 3. Pilot Test

Run with 5-10 participants to verify:
- Timing is appropriate
- Instructions are clear
- Data is recording correctly
- No technical issues

### 4. Power Analysis

With n=60 per condition:
- Power to detect interaction η²p = 0.06: 95%
- Power to detect within-condition asymmetry d = 0.8: >99%

## Analysis Pipeline

After data collection, analyze with:

```python
from src.analysis.experimental import ExperimentalAnalyzer

analyzer = ExperimentalAnalyzer()
analyzer.load_from_database("data/reversal_curse.db")
results = analyzer.run_full_analysis()

# Key outputs:
# - 2×3 Mixed ANOVA (Direction × Condition)
# - Within-condition paired t-tests
# - Effect sizes (Cohen's d, η²p)
# - The Flip test (asymmetry sign reversal)
```

## Expected Results

If the TCA hypothesis is correct:

| Condition | Forward Acc | Reverse Acc | Asymmetry |
|-----------|-------------|-------------|-----------|
| A-then-B | ~78% | ~58% | +20pp |
| B-then-A | ~56% | ~76% | −20pp |
| Simultaneous | ~68% | ~66% | +2pp (NS) |

The critical finding is the **sign reversal** of asymmetry between sequential conditions.

## Relation to AI Systems

The "reversal curse" was first identified in large language models (Berglund et al., 2023). This project investigates whether humans show a similar constraint.

**We do not claim humans "have the reversal curse like LLMs"** — mechanisms differ. We investigate whether both systems exhibit signatures of temporal credit assignment, though through different mechanisms:
- LLMs: Gradient-based weight updates
- Humans: Synaptic plasticity with eligibility traces

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@misc{temporalcreditassignment2025,
  title={Temporal Credit Assignment Shapes the Direction of Knowledge Retrieval},
  author={[Authors]},
  year={2025},
  note={Pre-registration and experimental platform}
}
```

## Contact

For questions about this research, please open an issue on GitHub.
