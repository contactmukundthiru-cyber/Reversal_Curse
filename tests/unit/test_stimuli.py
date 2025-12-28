"""
Unit tests for stimulus generation.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.experiment.stimuli import StimulusGenerator, Symbol, Label


class TestStimulusGenerator:
    """Tests for the stimulus generator."""

    def test_generate_symbols_count(self):
        """Should generate requested number of symbols."""
        generator = StimulusGenerator(seed=42)
        symbols = generator.generate_symbols(16)

        assert len(symbols) == 16

    def test_symbols_have_required_properties(self):
        """Each symbol should have all required properties."""
        generator = StimulusGenerator(seed=42)
        symbols = generator.generate_symbols(5)

        for symbol in symbols:
            assert hasattr(symbol, "id")
            assert hasattr(symbol, "svg_path")
            assert hasattr(symbol, "svg_viewbox")
            assert symbol.id is not None
            assert len(symbol.svg_path) > 10  # Non-trivial path

    def test_symbols_are_unique(self):
        """Generated symbols should be unique."""
        generator = StimulusGenerator(seed=42)
        symbols = generator.generate_symbols(16)

        ids = [s.id for s in symbols]
        assert len(ids) == len(set(ids))  # All unique IDs

    def test_generate_labels_count(self):
        """Should generate requested number of labels."""
        generator = StimulusGenerator(seed=42)
        labels = generator.generate_labels(16)

        assert len(labels) == 16

    def test_labels_are_valid(self):
        """Labels should meet length and format requirements."""
        generator = StimulusGenerator(seed=42)
        labels = generator.generate_labels(16)

        for label in labels:
            assert 5 <= len(label) <= 7
            assert label.isupper()
            assert label.isalpha()

    def test_labels_are_unique(self):
        """Generated labels should be unique."""
        generator = StimulusGenerator(seed=42)
        labels = generator.generate_labels(16)

        assert len(labels) == len(set(labels))

    def test_seed_reproducibility(self):
        """Same seed should produce same output."""
        gen1 = StimulusGenerator(seed=123)
        gen2 = StimulusGenerator(seed=123)

        symbols1 = gen1.generate_symbols(5)
        symbols2 = gen2.generate_symbols(5)

        for s1, s2 in zip(symbols1, symbols2):
            assert s1.svg_path == s2.svg_path

    def test_different_seeds_different_output(self):
        """Different seeds should produce different output."""
        gen1 = StimulusGenerator(seed=123)
        gen2 = StimulusGenerator(seed=456)

        symbols1 = gen1.generate_symbols(5)
        symbols2 = gen2.generate_symbols(5)

        # At least some should be different
        paths1 = [s.svg_path for s in symbols1]
        paths2 = [s.svg_path for s in symbols2]
        assert paths1 != paths2

    def test_generate_stimulus_set(self):
        """Should generate complete stimulus set with pairings."""
        generator = StimulusGenerator(seed=42)
        stimulus_set = generator.generate_stimulus_set(n_pairs=16)

        assert len(stimulus_set) == 16

        for symbol_id, stim in stimulus_set.items():
            assert "symbol" in stim
            assert "label" in stim
            assert stim["symbol"]["id"] == symbol_id
            assert len(stim["label"]) >= 5

    def test_validate_stimulus_set(self):
        """Should validate correct stimulus sets."""
        generator = StimulusGenerator(seed=42)
        stimulus_set = generator.generate_stimulus_set(n_pairs=16)

        is_valid, issues = generator.validate_stimulus_set(stimulus_set)
        assert is_valid
        assert len(issues) == 0

    def test_validate_catches_short_labels(self):
        """Validation should catch labels that are too short."""
        generator = StimulusGenerator(seed=42)
        stimulus_set = generator.generate_stimulus_set(n_pairs=16)

        # Corrupt one label
        first_key = list(stimulus_set.keys())[0]
        stimulus_set[first_key]["label"] = "AB"

        is_valid, issues = generator.validate_stimulus_set(stimulus_set)
        assert not is_valid
        assert any("length" in issue.lower() for issue in issues)


class TestSymbol:
    """Tests for Symbol dataclass."""

    def test_symbol_to_dict(self):
        """Should convert to dictionary correctly."""
        symbol = Symbol(
            id="test_01",
            svg_path="M 0 0 L 100 100",
            svg_viewbox="0 0 100 100"
        )

        d = symbol.to_dict()
        assert d["id"] == "test_01"
        assert d["svg_path"] == "M 0 0 L 100 100"

    def test_symbol_to_svg(self):
        """Should generate valid SVG string."""
        symbol = Symbol(
            id="test_01",
            svg_path="M 0 0 L 100 100"
        )

        svg = symbol.to_svg()
        assert "<svg" in svg
        assert "</svg>" in svg
        assert symbol.svg_path in svg


class TestLabel:
    """Tests for Label dataclass."""

    def test_label_to_dict(self):
        """Should convert to dictionary correctly."""
        label = Label(
            text="BLICKET",
            phonetic="BLIK-et",
            syllables=2
        )

        d = label.to_dict()
        assert d["text"] == "BLICKET"
        assert d["syllables"] == 2


class TestPronounceable:
    """Tests for pronounceability checking."""

    def test_pronounceable_with_vowels(self):
        """Words with vowels should be pronounceable."""
        generator = StimulusGenerator()

        assert generator._is_pronounceable("BLICKET")
        assert generator._is_pronounceable("ZORBIT")
        assert generator._is_pronounceable("DAXEN")

    def test_not_pronounceable_no_vowels(self):
        """Words without vowels should not be pronounceable."""
        generator = StimulusGenerator()

        assert not generator._is_pronounceable("BRRRT")
        assert not generator._is_pronounceable("XYZWQ")

    def test_not_pronounceable_too_many_consonants(self):
        """Words with too many consecutive consonants should fail."""
        generator = StimulusGenerator()

        assert not generator._is_pronounceable("STRNGTH")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
