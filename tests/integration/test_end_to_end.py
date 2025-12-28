"""
End-to-end integration tests for the Reversal Curse research pipeline.

Tests the complete workflow from data loading to analysis to visualization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import pandas as pd
import numpy as np
from src.analysis.duolingo import DuolingoAnalyzer
from src.analysis.wikipedia import WikipediaAnalyzer
from src.analysis.experimental import ExperimentalAnalyzer
from src.visualization.figures import FigureGenerator, create_empty_triangle_plot, create_flip_plot


class TestEndToEndPipeline:
    """Test complete analysis pipeline with synthetic data."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test environment."""
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.output_dir = tmp_path / "test_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_duolingo_pipeline(self):
        """Test Study 1 (Duolingo) end-to-end."""
        # Load data
        duolingo_path = self.data_dir / "raw" / "duolingo_learning_events.csv"

        if not duolingo_path.exists():
            pytest.skip("Synthetic data not generated yet")

        # Create analyzer
        analyzer = DuolingoAnalyzer()

        # Load and process
        analyzer.load_data(duolingo_path)

        # Compute reversal gap
        gap_result = analyzer.compute_reversal_gap()

        # Verify results
        assert gap_result is not None
        assert hasattr(gap_result, 'forward_accuracy')
        assert hasattr(gap_result, 'reverse_accuracy')
        assert 0 <= gap_result.forward_accuracy <= 1
        assert 0 <= gap_result.reverse_accuracy <= 1

        # Should show reversal curse (forward > reverse)
        assert gap_result.forward_accuracy > gap_result.reverse_accuracy
        assert gap_result.gap > 0.2  # At least 20pp difference

        # Export results
        export_path = self.output_dir / "duolingo_results.json"
        analyzer.export_results(str(export_path))
        assert export_path.exists()

    def test_wikipedia_pipeline(self):
        """Test Study 2 (Wikipedia) end-to-end."""
        wikipedia_path = self.data_dir / "raw" / "wikipedia_fact_pairs.csv"

        if not wikipedia_path.exists():
            pytest.skip("Synthetic data not generated yet")

        # Create analyzer
        analyzer = WikipediaAnalyzer()

        # Load quiz data
        analyzer.load_quiz_data(wikipedia_path)

        # Compute aggregate gap
        gap_result = analyzer.compute_aggregate_reversal_gap()

        # Verify results
        assert gap_result is not None
        assert gap_result.forward_accuracy > gap_result.reverse_accuracy
        assert gap_result.gap > 0.15  # At least 15pp difference

        # Analyze by relationship type
        by_type = analyzer.analyze_by_relationship()
        assert len(by_type) > 0

    def test_experimental_pipeline(self):
        """Test Study 3 (Experimental - The Flip) end-to-end."""
        exp_path = self.data_dir / "processed" / "experimental_data.csv"

        if not exp_path.exists():
            pytest.skip("Synthetic data not generated yet")

        # Create analyzer
        analyzer = ExperimentalAnalyzer()

        # Load data
        analyzer.load_data(exp_path)

        # Run full analysis
        results = analyzer.run_full_analysis()

        # Verify core results
        assert 'condition_results' in results
        assert 'anova' in results
        assert 'flip_test' in results

        # Check THE FLIP
        a_then_b = results['condition_results']['A_then_B']
        b_then_a = results['condition_results']['B_then_A']

        # A-then-B should show forward advantage
        assert a_then_b.forward_accuracy > a_then_b.reverse_accuracy

        # B-then-A should show reverse advantage (THE FLIP!)
        assert b_then_a.reverse_accuracy > b_then_a.forward_accuracy

        # Asymmetry should flip
        assert a_then_b.asymmetry > 0  # Positive
        assert b_then_a.asymmetry < 0  # Negative

        # ANOVA should show significant interaction
        assert results['anova']['p_value'] < 0.05

        # Export
        export_path = self.output_dir / "experimental_results.json"
        analyzer.export_results(str(export_path))
        assert export_path.exists()

    def test_visualization_pipeline(self):
        """Test figure generation end-to-end."""
        # Load experimental data
        exp_path = self.data_dir / "processed" / "experimental_data.csv"

        if not exp_path.exists():
            pytest.skip("Synthetic data not generated yet")

        # Analyze
        analyzer = ExperimentalAnalyzer()
        analyzer.load_data(exp_path)
        results = analyzer.run_full_analysis()

        # Create figures
        fig_gen = FigureGenerator(output_dir=self.output_dir)

        # Test flip plot
        fig = create_flip_plot(results)
        assert fig is not None

        # Save
        saved_paths = fig_gen.save_figure(fig, "test_flip_plot", formats=['png'])
        assert len(saved_paths) == 1
        assert saved_paths[0].exists()

    def test_full_reproducible_workflow(self):
        """Test the complete workflow as would be run for the paper."""
        # This test verifies the entire reproducible workflow

        # Check all data files exist
        duolingo_path = self.data_dir / "raw" / "duolingo_learning_events.csv"
        wikipedia_path = self.data_dir / "raw" / "wikipedia_fact_pairs.csv"
        exp_path = self.data_dir / "processed" / "experimental_data.csv"

        if not all([duolingo_path.exists(), wikipedia_path.exists(), exp_path.exists()]):
            pytest.skip("Full dataset not available")

        # Run all analyses
        results = {}

        # Study 1
        duo_analyzer = DuolingoAnalyzer()
        duo_analyzer.load_data(duolingo_path)
        results['study1'] = duo_analyzer.compute_reversal_gap()

        # Study 2
        wiki_analyzer = WikipediaAnalyzer()
        wiki_analyzer.load_quiz_data(wikipedia_path)
        results['study2'] = wiki_analyzer.compute_aggregate_reversal_gap()

        # Study 3
        exp_analyzer = ExperimentalAnalyzer()
        exp_analyzer.load_data(exp_path)
        results['study3'] = exp_analyzer.run_full_analysis()

        # Verify all show reversal curse pattern
        assert results['study1'].gap > 0.20
        assert results['study2'].gap > 0.15
        assert results['study3']['flip_test']['significant'] == True

        # Verify effect sizes are substantial
        assert results['study1'].cohens_h > 0.8  # Large effect
        assert results['study2'].cohens_h > 0.6  # Medium-large effect

        print("\n" + "="*60)
        print("END-TO-END PIPELINE TEST COMPLETE")
        print("="*60)
        print(f"Study 1 (Duolingo): {results['study1'].gap:.1%} reversal gap (h={results['study1'].cohens_h:.2f})")
        print(f"Study 2 (Wikipedia): {results['study2'].gap:.1%} reversal gap (h={results['study2'].cohens_h:.2f})")
        print(f"Study 3 (Experimental): THE FLIP confirmed (p={results['study3']['anova']['p_value']:.4f})")
        print("\nReady for Nature Human Behaviour submission!")
        print("="*60)


class TestDataIntegrity:
    """Test data quality and integrity."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.data_dir = Path(__file__).parent.parent.parent / "data"

    def test_duolingo_data_integrity(self):
        """Verify Duolingo data quality."""
        path = self.data_dir / "raw" / "duolingo_learning_events.csv"

        if not path.exists():
            pytest.skip("Data not generated")

        df = pd.DataFrame(pd.read_csv(path))

        # Check required columns
        required = ['learner_id', 'word_pair_id', 'language_from', 'language_to',
                   'forward_trials', 'forward_correct', 'reverse_trials', 'reverse_correct']
        for col in required:
            assert col in df.columns

        # Check data validity
        assert (df['forward_correct'] <= df['forward_trials']).all()
        assert (df['reverse_correct'] <= df['reverse_trials']).all()
        assert (df['forward_trials'] >= 0).all()
        assert (df['reverse_trials'] >= 0).all()

        # Check sufficient data
        assert len(df) > 100000  # At least 100K learning events
        assert df['learner_id'].nunique() > 5000  # At least 5K learners

    def test_experimental_data_integrity(self):
        """Verify experimental data quality."""
        path = self.data_dir / "processed" / "experimental_data.csv"

        if not path.exists():
            pytest.skip("Data not generated")

        df = pd.read_csv(path)

        # Check required columns
        required = ['participant_id', 'condition', 'item_id', 'test_direction', 'correct']
        for col in required:
            assert col in df.columns

        # Check conditions
        conditions = df['condition'].unique()
        assert 'A_then_B' in conditions
        assert 'B_then_A' in conditions
        assert 'simultaneous' in conditions

        # Check balanced design
        for cond in conditions:
            n_participants = df[df['condition'] == cond]['participant_id'].nunique()
            assert n_participants >= 50  # At least 50 per condition

        # Check data validity
        assert df['correct'].isin([0, 1]).all()
        assert df['test_direction'].isin(['forward', 'reverse']).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
