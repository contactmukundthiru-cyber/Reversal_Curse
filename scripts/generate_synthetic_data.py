#!/usr/bin/env python3
"""
Generate DEMO/TEST data for development and testing ONLY.

WARNING: This script generates SYNTHETIC data for testing the analysis pipeline.
This data is NOT real experimental data and MUST NOT be used for:
- Scientific publications
- Drawing conclusions about the hypothesis
- Presenting as actual research results

Use this only for:
- Testing that the analysis code works correctly
- Verifying data format compatibility
- Development and debugging
- Demo purposes

For actual research, you must:
1. Obtain IRB/ethics approval
2. Recruit real participants
3. Collect genuine experimental data
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
import json
import warnings

# Issue a warning every time this is run
warnings.warn(
    "\n\n"
    "=" * 60 + "\n"
    "WARNING: GENERATING SYNTHETIC DEMO DATA\n"
    "This is NOT real experimental data!\n"
    "Do NOT use for publications or research conclusions.\n"
    "=" * 60 + "\n",
    UserWarning
)


def generate_demo_experimental_data(n_per_condition: int = 10) -> pd.DataFrame:
    """
    Generate a small demo dataset for testing the analysis pipeline.

    This uses arbitrary probability distributions and is NOT meant to
    represent real experimental findings.

    Parameters
    ----------
    n_per_condition : int
        Number of simulated participants per condition (keep small for demos)

    Returns
    -------
    pd.DataFrame
        Demo data in the expected format
    """
    print(f"Generating DEMO data: {n_per_condition} participants per condition...")
    print("WARNING: This is fake data for testing only!")

    rng = np.random.default_rng(12345)  # Fixed seed for reproducibility

    conditions = ["A_then_B", "B_then_A", "simultaneous"]
    n_test_items = 12

    records = []

    for condition in conditions:
        for participant_id in range(n_per_condition):
            pid = f"DEMO_{condition}_{participant_id:03d}"

            # Random performance (no systematic pattern - just testing data format)
            base_memory = rng.beta(5, 3)

            for item_id in range(n_test_items):
                # Forward test
                forward_correct = rng.random() < (base_memory * 0.7 + 0.15)
                forward_rt = rng.lognormal(7.2, 0.4)

                records.append({
                    "participant_id": pid,
                    "condition": condition,
                    "item_id": item_id,
                    "test_direction": "forward",
                    "correct": int(forward_correct),
                    "reaction_time_ms": float(forward_rt),
                    "confidence": int(rng.integers(1, 6))
                })

                # Reverse test
                reverse_correct = rng.random() < (base_memory * 0.7 + 0.15)
                reverse_rt = rng.lognormal(7.3, 0.4)

                records.append({
                    "participant_id": pid,
                    "condition": condition,
                    "item_id": item_id,
                    "test_direction": "reverse",
                    "correct": int(reverse_correct),
                    "reaction_time_ms": float(reverse_rt),
                    "confidence": int(rng.integers(1, 6))
                })

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} DEMO trials from {n_per_condition * 3} fake participants")
    return df


def generate_demo_data():
    """Generate demo data and save with clear warnings."""
    data_dir = Path(__file__).parent.parent / "data"

    # Create directories
    (data_dir / "demo").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING DEMO DATA FOR TESTING ONLY")
    print("This is NOT real experimental data!")
    print("=" * 60 + "\n")

    # Generate demo data
    demo_df = generate_demo_experimental_data(n_per_condition=10)

    # Save with clear naming
    demo_path = data_dir / "demo" / "DEMO_DATA_NOT_REAL.csv"
    demo_df.to_csv(demo_path, index=False)
    print(f"\nSaved: {demo_path}")

    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "WARNING": "THIS IS SYNTHETIC DEMO DATA - NOT REAL EXPERIMENTAL DATA",
        "purpose": "Testing analysis pipeline only",
        "do_not_use_for": [
            "Scientific publications",
            "Research conclusions",
            "ISEF or other competitions",
            "Any form of real research"
        ],
        "n_participants": 30,
        "n_trials": len(demo_df)
    }

    metadata_path = data_dir / "demo" / "DEMO_METADATA.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {metadata_path}")

    print("\n" + "=" * 60)
    print("DEMO DATA GENERATED")
    print("Remember: This is for TESTING ONLY, not real research!")
    print("=" * 60)


if __name__ == "__main__":
    generate_demo_data()
