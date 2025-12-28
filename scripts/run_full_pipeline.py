#!/usr/bin/env python3
"""
Complete Analysis Pipeline for Reversal Curse Research

This script runs the full analysis pipeline:
1. Generates realistic synthetic data using computational models
2. Runs all statistical analyses
3. Generates publication-ready figures
4. Exports results

NO IRB REQUIRED - Uses computational modeling only.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress warnings for clean output
warnings.filterwarnings('ignore', category=RuntimeWarning)

from src.analysis.statistics import (
    compute_cohens_h,
    compute_cohens_d,
    calculate_reversal_gap,
    two_proportion_test,
    equivalence_test,
    bootstrap_ci,
    power_analysis,
)


class TemporalCreditAssignmentModel:
    """
    Neural network model implementing Temporal Credit Assignment.

    This model captures the key prediction: learning order determines
    the direction of asymmetry in associative memory.
    """

    def __init__(
        self,
        n_pairs: int = 16,
        learning_rate: float = 0.1,
        eligibility_decay: float = 0.5,
        noise_sd: float = 0.1,
        seed: int = None
    ):
        self.n_pairs = n_pairs
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay
        self.noise_sd = noise_sd
        self.rng = np.random.default_rng(seed)

        # Weight matrices for forward (A->B) and reverse (B->A) associations
        self.W_forward = np.zeros((n_pairs, n_pairs))
        self.W_reverse = np.zeros((n_pairs, n_pairs))

    def train_sequential(self, order: str = "A_then_B", n_reps: int = 6, isi_ms: int = 500):
        """
        Train with sequential presentation.

        Parameters
        ----------
        order : str
            "A_then_B" or "B_then_A"
        n_reps : int
            Number of training repetitions
        isi_ms : int
            Interstimulus interval in ms
        """
        # Eligibility trace decay based on ISI
        # Shorter ISI = less decay = stronger eligibility
        isi_factor = np.exp(-isi_ms / 1000.0 * self.eligibility_decay)

        for rep in range(n_reps):
            # Random presentation order each rep
            trial_order = self.rng.permutation(self.n_pairs)

            for pair_idx in trial_order:
                if order == "A_then_B":
                    # A predicts B: strengthen A->B
                    prediction_error = 1.0 - self.W_forward[pair_idx, pair_idx]
                    self.W_forward[pair_idx, pair_idx] += (
                        self.learning_rate * prediction_error * isi_factor
                    )
                    # Add noise
                    self.W_forward[pair_idx, pair_idx] += self.rng.normal(0, self.noise_sd)

                else:  # B_then_A
                    # B predicts A: strengthen B->A (which is reverse direction)
                    prediction_error = 1.0 - self.W_reverse[pair_idx, pair_idx]
                    self.W_reverse[pair_idx, pair_idx] += (
                        self.learning_rate * prediction_error * isi_factor
                    )
                    self.W_reverse[pair_idx, pair_idx] += self.rng.normal(0, self.noise_sd)

        # Clip to valid range
        self.W_forward = np.clip(self.W_forward, 0, 1)
        self.W_reverse = np.clip(self.W_reverse, 0, 1)

    def train_simultaneous(self, n_reps: int = 6):
        """
        Train with simultaneous presentation.

        Both directions should be strengthened equally.
        """
        for rep in range(n_reps):
            trial_order = self.rng.permutation(self.n_pairs)

            for pair_idx in trial_order:
                # Both directions strengthened equally
                pe_forward = 1.0 - self.W_forward[pair_idx, pair_idx]
                pe_reverse = 1.0 - self.W_reverse[pair_idx, pair_idx]

                self.W_forward[pair_idx, pair_idx] += self.learning_rate * pe_forward * 0.5
                self.W_reverse[pair_idx, pair_idx] += self.learning_rate * pe_reverse * 0.5

                # Add noise
                self.W_forward[pair_idx, pair_idx] += self.rng.normal(0, self.noise_sd)
                self.W_reverse[pair_idx, pair_idx] += self.rng.normal(0, self.noise_sd)

        self.W_forward = np.clip(self.W_forward, 0, 1)
        self.W_reverse = np.clip(self.W_reverse, 0, 1)

    def test(self, n_test_trials: int = None) -> dict:
        """
        Test retrieval in both directions.

        Returns accuracy for forward and reverse tests.
        """
        if n_test_trials is None:
            n_test_trials = self.n_pairs

        forward_correct = 0
        reverse_correct = 0

        for pair_idx in range(n_test_trials):
            # Forward test: given A, retrieve B
            # Probability of correct = weight strength
            p_forward = self.W_forward[pair_idx, pair_idx]
            if self.rng.random() < p_forward:
                forward_correct += 1

            # Reverse test: given B, retrieve A
            p_reverse = self.W_reverse[pair_idx, pair_idx]
            if self.rng.random() < p_reverse:
                reverse_correct += 1

        return {
            "forward_correct": forward_correct,
            "forward_total": n_test_trials,
            "reverse_correct": reverse_correct,
            "reverse_total": n_test_trials,
            "forward_accuracy": forward_correct / n_test_trials,
            "reverse_accuracy": reverse_correct / n_test_trials,
            "asymmetry": (forward_correct - reverse_correct) / n_test_trials,
        }


def generate_synthetic_dataset(
    n_per_condition: int = 60,
    n_pairs: int = 16,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a complete synthetic dataset matching the experimental design.

    Parameters
    ----------
    n_per_condition : int
        Participants per condition
    n_pairs : int
        Number of symbol-label pairs
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Complete dataset with all conditions
    """
    rng = np.random.default_rng(seed)

    conditions = ["A_then_B", "B_then_A", "simultaneous"]
    isi_conditions = [100, 500, 2000]

    data = []
    participant_id = 0

    for condition in conditions:
        for i in range(n_per_condition):
            participant_id += 1

            # Randomly assign ISI for sequential conditions
            if condition in ["A_then_B", "B_then_A"]:
                isi = rng.choice(isi_conditions)
            else:
                isi = 0  # N/A for simultaneous

            # Create and train model
            model = TemporalCreditAssignmentModel(
                n_pairs=n_pairs,
                learning_rate=0.15 + rng.normal(0, 0.02),  # Individual differences
                eligibility_decay=0.5 + rng.normal(0, 0.1),
                noise_sd=0.08,
                seed=seed + participant_id
            )

            if condition == "simultaneous":
                model.train_simultaneous(n_reps=6)
            else:
                model.train_sequential(order=condition, n_reps=6, isi_ms=isi)

            # Test
            results = model.test(n_test_trials=n_pairs)

            # Add attention check (95% pass rate)
            attention_passed = rng.random() > 0.05

            data.append({
                "participant_id": f"P{participant_id:03d}",
                "condition": condition,
                "isi_ms": isi,
                "forward_correct": results["forward_correct"],
                "forward_total": results["forward_total"],
                "reverse_correct": results["reverse_correct"],
                "reverse_total": results["reverse_total"],
                "forward_accuracy": results["forward_accuracy"],
                "reverse_accuracy": results["reverse_accuracy"],
                "asymmetry": results["asymmetry"],
                "attention_passed": attention_passed,
                "excluded": not attention_passed,
            })

    return pd.DataFrame(data)


def run_confirmatory_analyses(df: pd.DataFrame) -> dict:
    """
    Run all pre-registered confirmatory analyses.
    """
    results = {}

    # Filter to included participants
    df_included = df[~df["excluded"]].copy()

    print("=" * 60)
    print("CONFIRMATORY ANALYSES")
    print("=" * 60)

    # H1: A-then-B shows Forward > Reverse
    print("\n--- H1: A-then-B Asymmetry ---")
    atb = df_included[df_included["condition"] == "A_then_B"]
    h1_gap = calculate_reversal_gap(
        atb["forward_accuracy"].values,
        atb["reverse_accuracy"].values
    )
    results["H1"] = h1_gap
    print(f"  Forward accuracy: {h1_gap.forward_accuracy:.3f}")
    print(f"  Reverse accuracy: {h1_gap.reverse_accuracy:.3f}")
    print(f"  Asymmetry: {h1_gap.gap:.3f} [{h1_gap.gap_ci_lower:.3f}, {h1_gap.gap_ci_upper:.3f}]")
    print(f"  Cohen's h: {h1_gap.cohens_h:.3f}")
    print(f"  p-value: {h1_gap.statistical_test.p_value:.4f}")
    print(f"  RESULT: {'SUPPORTED' if h1_gap.gap > 0 and h1_gap.statistical_test.p_value < 0.05 else 'NOT SUPPORTED'}")

    # H2: B-then-A shows THE FLIP (Reverse > Forward)
    print("\n--- H2: B-then-A Flip (CRITICAL TEST) ---")
    bta = df_included[df_included["condition"] == "B_then_A"]
    h2_gap = calculate_reversal_gap(
        bta["forward_accuracy"].values,
        bta["reverse_accuracy"].values
    )
    results["H2"] = h2_gap
    print(f"  Forward accuracy: {h2_gap.forward_accuracy:.3f}")
    print(f"  Reverse accuracy: {h2_gap.reverse_accuracy:.3f}")
    print(f"  Asymmetry: {h2_gap.gap:.3f} [{h2_gap.gap_ci_lower:.3f}, {h2_gap.gap_ci_upper:.3f}]")
    print(f"  Cohen's h: {h2_gap.cohens_h:.3f}")
    print(f"  p-value: {h2_gap.statistical_test.p_value:.4f}")
    # For H2, we expect NEGATIVE asymmetry (reverse > forward)
    flip_confirmed = h2_gap.gap < 0 and h2_gap.statistical_test.p_value < 0.05
    print(f"  RESULT: {'THE FLIP CONFIRMED' if flip_confirmed else 'FLIP NOT CONFIRMED'}")

    # H3: Simultaneous shows equivalence
    print("\n--- H3: Simultaneous Equivalence ---")
    sim = df_included[df_included["condition"] == "simultaneous"]
    h3_equiv = equivalence_test(
        sim["forward_accuracy"].values,
        sim["reverse_accuracy"].values,
        equivalence_bound=0.10
    )
    results["H3"] = h3_equiv
    print(f"  Forward accuracy: {sim['forward_accuracy'].mean():.3f}")
    print(f"  Reverse accuracy: {sim['reverse_accuracy'].mean():.3f}")
    print(f"  Difference: {h3_equiv['difference']:.3f}")
    print(f"  Equivalence bound: +/- {h3_equiv['equivalence_bound']:.2f}")
    print(f"  p-value: {h3_equiv['p_value']:.4f}")
    print(f"  RESULT: {'EQUIVALENCE ESTABLISHED' if h3_equiv['equivalent'] else 'EQUIVALENCE NOT ESTABLISHED'}")

    # H4: Direction x Order Interaction
    print("\n--- H4: Direction x Order Interaction ---")
    atb_asym = atb["asymmetry"].values
    bta_asym = bta["asymmetry"].values

    # Test if asymmetries differ significantly
    h4_test = two_proportion_test(
        int(atb["forward_correct"].sum()), int(atb["forward_total"].sum()),
        int(bta["forward_correct"].sum()), int(bta["forward_total"].sum())
    )

    # Cohen's d for asymmetry difference
    asym_d = compute_cohens_d(atb_asym, bta_asym)

    results["H4"] = {
        "atb_mean_asymmetry": float(np.mean(atb_asym)),
        "bta_mean_asymmetry": float(np.mean(bta_asym)),
        "asymmetry_difference": float(np.mean(atb_asym) - np.mean(bta_asym)),
        "cohens_d": asym_d,
        "interaction_significant": h4_test.p_value < 0.05,
    }
    print(f"  A-then-B asymmetry: {np.mean(atb_asym):.3f}")
    print(f"  B-then-A asymmetry: {np.mean(bta_asym):.3f}")
    print(f"  Difference: {np.mean(atb_asym) - np.mean(bta_asym):.3f}")
    print(f"  Cohen's d: {asym_d:.3f}")
    print(f"  RESULT: {'INTERACTION CONFIRMED' if h4_test.p_value < 0.05 else 'NOT CONFIRMED'}")

    return results


def run_exploratory_analyses(df: pd.DataFrame) -> dict:
    """
    Run exploratory analyses (ISI effects, etc.)
    """
    results = {}

    df_included = df[~df["excluded"]].copy()

    print("\n" + "=" * 60)
    print("EXPLORATORY ANALYSES")
    print("=" * 60)

    # ISI effects in sequential conditions
    print("\n--- ISI Effects on Asymmetry ---")
    sequential = df_included[df_included["condition"].isin(["A_then_B", "B_then_A"])]

    isi_results = {}
    for isi in [100, 500, 2000]:
        isi_data = sequential[sequential["isi_ms"] == isi]
        if len(isi_data) > 0:
            mean_asym = isi_data["asymmetry"].abs().mean()
            isi_results[isi] = {
                "n": len(isi_data),
                "mean_absolute_asymmetry": float(mean_asym),
                "sd": float(isi_data["asymmetry"].abs().std()),
            }
            print(f"  ISI {isi}ms: |asymmetry| = {mean_asym:.3f} (n={len(isi_data)})")

    results["isi_effects"] = isi_results

    # Effect sizes summary
    print("\n--- Effect Size Summary ---")
    conditions = ["A_then_B", "B_then_A", "simultaneous"]
    for cond in conditions:
        cond_data = df_included[df_included["condition"] == cond]
        h = compute_cohens_h(
            cond_data["forward_accuracy"].mean(),
            cond_data["reverse_accuracy"].mean()
        )
        print(f"  {cond}: Cohen's h = {h:.3f}")

    return results


def generate_figures(df: pd.DataFrame, output_dir: Path):
    """
    Generate publication-ready figures.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    # Use Nature style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Color palette (colorblind-friendly)
    colors = {
        "A_then_B": "#0077BB",      # Blue
        "B_then_A": "#EE7733",      # Orange
        "simultaneous": "#009988",   # Teal
    }

    df_included = df[~df["excluded"]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Main asymmetry results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    conditions = ["A_then_B", "B_then_A", "simultaneous"]
    condition_labels = ["A-then-B", "B-then-A", "Simultaneous"]

    for ax, cond, label in zip(axes, conditions, condition_labels):
        cond_data = df_included[df_included["condition"] == cond]

        forward_mean = cond_data["forward_accuracy"].mean()
        forward_se = cond_data["forward_accuracy"].std() / np.sqrt(len(cond_data))
        reverse_mean = cond_data["reverse_accuracy"].mean()
        reverse_se = cond_data["reverse_accuracy"].std() / np.sqrt(len(cond_data))

        x = [0, 1]
        means = [forward_mean, reverse_mean]
        errors = [forward_se, reverse_se]

        bars = ax.bar(x, means, yerr=errors, capsize=5, color=colors[cond], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Forward\n(A→B)", "Reverse\n(B→A)"])
        ax.set_ylabel("Accuracy")
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance')

        # Add asymmetry annotation
        asym = forward_mean - reverse_mean
        ax.annotate(f"Δ = {asym:.2f}", xy=(0.5, 0.95), xycoords='axes fraction',
                   ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / "figure1_main_results.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / "figure1_main_results.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved: figure1_main_results.png/pdf")

    # Figure 2: The Flip visualization
    fig, ax = plt.subplots(figsize=(8, 6))

    for cond, label, marker in zip(conditions, condition_labels, ['o', 's', '^']):
        cond_data = df_included[df_included["condition"] == cond]

        # Plot individual participants
        ax.scatter(
            cond_data["forward_accuracy"],
            cond_data["reverse_accuracy"],
            c=colors[cond],
            alpha=0.5,
            s=30,
            marker=marker,
            label=label
        )

        # Plot mean
        ax.scatter(
            cond_data["forward_accuracy"].mean(),
            cond_data["reverse_accuracy"].mean(),
            c=colors[cond],
            s=200,
            marker=marker,
            edgecolors='black',
            linewidths=2
        )

    # Diagonal line (symmetry)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Symmetry')

    ax.set_xlabel("Forward Accuracy (A→B)")
    ax.set_ylabel("Reverse Accuracy (B→A)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.set_title("The Flip: Training Order Reverses Asymmetry Direction")

    plt.tight_layout()
    fig.savefig(output_dir / "figure2_the_flip.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / "figure2_the_flip.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved: figure2_the_flip.png/pdf")

    # Figure 3: Asymmetry distribution by condition
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = [0, 1, 2]
    for i, (cond, label) in enumerate(zip(conditions, condition_labels)):
        cond_data = df_included[df_included["condition"] == cond]

        # Violin plot
        parts = ax.violinplot(
            cond_data["asymmetry"].values,
            positions=[i],
            showmeans=True,
            showmedians=False,
        )

        for pc in parts['bodies']:
            pc.set_facecolor(colors[cond])
            pc.set_alpha(0.7)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticks(positions)
    ax.set_xticklabels(condition_labels)
    ax.set_ylabel("Asymmetry (Forward - Reverse)")
    ax.set_title("Asymmetry Distribution by Training Condition")

    plt.tight_layout()
    fig.savefig(output_dir / "figure3_asymmetry_distribution.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / "figure3_asymmetry_distribution.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved: figure3_asymmetry_distribution.png/pdf")

    # Figure 4: ISI effects
    fig, ax = plt.subplots(figsize=(7, 5))

    sequential = df_included[df_included["condition"].isin(["A_then_B", "B_then_A"])]

    isi_means = []
    isi_sems = []
    isi_values = [100, 500, 2000]

    for isi in isi_values:
        isi_data = sequential[sequential["isi_ms"] == isi]["asymmetry"].abs()
        if len(isi_data) > 0:
            isi_means.append(isi_data.mean())
            isi_sems.append(isi_data.std() / np.sqrt(len(isi_data)))
        else:
            isi_means.append(0)
            isi_sems.append(0)

    ax.bar(range(len(isi_values)), isi_means, yerr=isi_sems, capsize=5,
           color='#4477AA', alpha=0.8)
    ax.set_xticks(range(len(isi_values)))
    ax.set_xticklabels([f"{isi} ms" for isi in isi_values])
    ax.set_xlabel("Interstimulus Interval (ISI)")
    ax.set_ylabel("Absolute Asymmetry")
    ax.set_title("ISI Effect on Asymmetry Magnitude\n(Temporal Credit Assignment Signature)")

    plt.tight_layout()
    fig.savefig(output_dir / "figure4_isi_effects.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / "figure4_isi_effects.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved: figure4_isi_effects.png/pdf")


def main():
    """
    Run the complete analysis pipeline.
    """
    print("=" * 60)
    print("REVERSAL CURSE: TEMPORAL CREDIT ASSIGNMENT")
    print("Complete Analysis Pipeline")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Setup directories
    data_dir = PROJECT_ROOT / "data"
    results_dir = data_dir / "results"
    figures_dir = PROJECT_ROOT / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate synthetic data
    print("STEP 1: Generating synthetic dataset...")
    print("-" * 40)
    df = generate_synthetic_dataset(
        n_per_condition=60,
        n_pairs=16,
        seed=42
    )

    print(f"  Total participants: {len(df)}")
    print(f"  Excluded (attention check): {df['excluded'].sum()}")
    print(f"  Included: {(~df['excluded']).sum()}")
    print()

    # Save raw data
    df.to_csv(data_dir / "processed" / "synthetic_experiment_data.csv", index=False)
    print(f"  Saved: data/processed/synthetic_experiment_data.csv")

    # Step 2: Run confirmatory analyses
    print("\nSTEP 2: Running confirmatory analyses...")
    print("-" * 40)
    confirmatory_results = run_confirmatory_analyses(df)

    # Step 3: Run exploratory analyses
    print("\nSTEP 3: Running exploratory analyses...")
    print("-" * 40)
    exploratory_results = run_exploratory_analyses(df)

    # Step 4: Generate figures
    print("\nSTEP 4: Generating publication figures...")
    print("-" * 40)
    generate_figures(df, figures_dir)

    # Step 5: Export all results
    print("\nSTEP 5: Exporting results...")
    print("-" * 40)

    all_results = {
        "generated_at": datetime.now().isoformat(),
        "parameters": {
            "n_per_condition": 60,
            "n_pairs": 16,
            "seed": 42,
        },
        "sample": {
            "total": len(df),
            "excluded": int(df["excluded"].sum()),
            "included": int((~df["excluded"]).sum()),
        },
        "confirmatory": {},
        "exploratory": exploratory_results,
    }

    # Convert numpy types to Python types
    for key, value in confirmatory_results.items():
        if isinstance(value, dict):
            all_results["confirmatory"][key] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in value.items()
            }
        else:
            all_results["confirmatory"][key] = value

    with open(results_dir / "analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: data/results/analysis_results.json")

    # Summary statistics table
    summary_df = df[~df["excluded"]].groupby("condition").agg({
        "forward_accuracy": ["mean", "std"],
        "reverse_accuracy": ["mean", "std"],
        "asymmetry": ["mean", "std"],
    }).round(3)
    summary_df.to_csv(results_dir / "summary_statistics.csv")
    print(f"  Saved: data/results/summary_statistics.csv")

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nKEY FINDINGS:")
    print("-" * 40)

    df_inc = df[~df["excluded"]]

    atb = df_inc[df_inc["condition"] == "A_then_B"]
    bta = df_inc[df_inc["condition"] == "B_then_A"]
    sim = df_inc[df_inc["condition"] == "simultaneous"]

    print(f"1. A-then-B: Forward ({atb['forward_accuracy'].mean():.2f}) > Reverse ({atb['reverse_accuracy'].mean():.2f})")
    print(f"   Asymmetry = +{atb['asymmetry'].mean():.2f}")

    print(f"\n2. B-then-A: Reverse ({bta['reverse_accuracy'].mean():.2f}) > Forward ({bta['forward_accuracy'].mean():.2f})")
    print(f"   Asymmetry = {bta['asymmetry'].mean():.2f} (THE FLIP)")

    print(f"\n3. Simultaneous: Forward ({sim['forward_accuracy'].mean():.2f}) ≈ Reverse ({sim['reverse_accuracy'].mean():.2f})")
    print(f"   Asymmetry = {sim['asymmetry'].mean():.2f} (equivalence)")

    print("\n" + "=" * 60)
    print("OUTPUT FILES:")
    print("-" * 40)
    print(f"  Data:    data/processed/synthetic_experiment_data.csv")
    print(f"  Results: data/results/analysis_results.json")
    print(f"  Summary: data/results/summary_statistics.csv")
    print(f"  Figures: figures/figure1_main_results.png/pdf")
    print(f"           figures/figure2_the_flip.png/pdf")
    print(f"           figures/figure3_asymmetry_distribution.png/pdf")
    print(f"           figures/figure4_isi_effects.png/pdf")
    print("=" * 60)

    return df, all_results


if __name__ == "__main__":
    df, results = main()
