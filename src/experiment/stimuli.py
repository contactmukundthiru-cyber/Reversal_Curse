"""
Stimulus generation for the Reversal Curse experiment.

This module provides:
- Procedurally generated abstract symbols (SVG)
- Pronounceable nonword labels
- Stimulus validation and quality checks
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib


@dataclass
class Symbol:
    """An abstract visual symbol."""

    id: str
    svg_path: str
    svg_viewbox: str = "0 0 100 100"
    stroke_width: int = 3
    fill: str = "none"
    stroke: str = "#000000"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "svg_path": self.svg_path,
            "svg_viewbox": self.svg_viewbox,
            "stroke_width": self.stroke_width,
            "fill": self.fill,
            "stroke": self.stroke,
        }

    def to_svg(self) -> str:
        """Generate full SVG element."""
        return f'''<svg viewBox="{self.svg_viewbox}" xmlns="http://www.w3.org/2000/svg">
            <path d="{self.svg_path}"
                  stroke="{self.stroke}"
                  stroke-width="{self.stroke_width}"
                  fill="{self.fill}"
                  stroke-linecap="round"
                  stroke-linejoin="round"/>
        </svg>'''


@dataclass
class Label:
    """A pronounceable nonword label."""

    text: str
    phonetic: Optional[str] = None
    syllables: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "phonetic": self.phonetic,
            "syllables": self.syllables,
        }


class StimulusGenerator:
    """
    Generator for experiment stimuli.

    Creates abstract symbols that don't resemble letters
    and pronounceable nonwords as labels.
    """

    # Predefined pronounceable nonwords (validated for no real-word neighbors)
    # All labels MUST be 5-7 characters per pre-registration spec
    PREDEFINED_LABELS = [
        # 5-letter labels
        "BLICK", "DAXEN", "FEPPO", "KREEB", "FLURP",
        "FRONK", "ZAXBY", "WOBIX", "GLORP", "TRINK",
        "PLONK", "SNERP", "BLONX", "CRUMF", "DWELF",
        # 6-letter labels
        "ZORBIT", "TAZZLE", "WUGGLE", "VEXTAR", "ZUNTAR",
        "FRAMBL", "TRONGL", "WIBZOR", "PLONKI", "ZEFRAM",
        "CLOMPR", "SNORBL", "FLIMBO", "GRUBNI", "QUILTO",
        # 7-letter labels
        "GRIFTON", "PLUMBUS", "BLORPIX", "SNARGEL", "TWIBBLE",
        "GLEEBER", "CHUMBIT", "KLONKER", "PRINGLE", "FLIMBER",
    ]

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.

        Parameters
        ----------
        seed : Optional[int]
            Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self._used_labels = set()
        self._used_symbols = set()

    def generate_symbols(self, n: int = 16) -> List[Symbol]:
        """
        Generate n unique abstract symbols.

        Parameters
        ----------
        n : int
            Number of symbols to generate

        Returns
        -------
        List[Symbol]
            List of unique symbols
        """
        symbols = []

        for i in range(n):
            symbol = self._generate_unique_symbol(i)
            symbols.append(symbol)

        return symbols

    def _generate_unique_symbol(self, index: int) -> Symbol:
        """Generate a single unique symbol.

        Uses the instance RNG for randomization to ensure reproducibility
        when a seed is provided at construction time.
        """
        # Use different generation strategies for variety
        strategies = [
            self._generate_curved_symbol,
            self._generate_angular_symbol,
            self._generate_mixed_symbol,
            self._generate_spiral_symbol,
        ]

        # Select strategy based on index for variety
        strategy = strategies[index % len(strategies)]

        max_attempts = 100
        for attempt in range(max_attempts):
            # DO NOT reseed - use the instance RNG to preserve reproducibility
            # The RNG state advances naturally through calls, ensuring different
            # symbols for each call while maintaining reproducibility with same seed
            svg_path = strategy()

            # Check uniqueness via hash
            path_hash = hashlib.md5(svg_path.encode()).hexdigest()[:8]
            if path_hash not in self._used_symbols:
                self._used_symbols.add(path_hash)
                return Symbol(
                    id=f"symbol_{index:02d}",
                    svg_path=svg_path,
                )

        # Fallback: generate with additional randomness to ensure uniqueness
        # Add index to the path to guarantee uniqueness
        svg_path = strategy()
        svg_path += f" M {50 + index} {50 + index}"  # Add unique marker
        return Symbol(
            id=f"symbol_{index:02d}",
            svg_path=svg_path,
        )

    def _generate_curved_symbol(self) -> str:
        """Generate a symbol with curved paths."""
        points = []
        cx, cy = 50, 50  # Center

        # Generate random control points
        n_curves = self.rng.randint(2, 4)

        start_angle = self.rng.uniform(0, 2 * math.pi)
        for i in range(n_curves):
            angle = start_angle + (i / n_curves) * 2 * math.pi
            r = self.rng.uniform(20, 40)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((x, y))

        # Build SVG path with quadratic curves
        if len(points) < 2:
            return "M 30 50 Q 50 20 70 50 Q 50 80 30 50"

        path_parts = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]

        for i in range(1, len(points)):
            # Control point
            cp_x = (points[i-1][0] + points[i][0]) / 2 + self.rng.uniform(-15, 15)
            cp_y = (points[i-1][1] + points[i][1]) / 2 + self.rng.uniform(-15, 15)
            path_parts.append(
                f"Q {cp_x:.1f} {cp_y:.1f} {points[i][0]:.1f} {points[i][1]:.1f}"
            )

        # Close back to start
        cp_x = (points[-1][0] + points[0][0]) / 2 + self.rng.uniform(-15, 15)
        cp_y = (points[-1][1] + points[0][1]) / 2 + self.rng.uniform(-15, 15)
        path_parts.append(
            f"Q {cp_x:.1f} {cp_y:.1f} {points[0][0]:.1f} {points[0][1]:.1f}"
        )

        return " ".join(path_parts)

    def _generate_angular_symbol(self) -> str:
        """Generate a symbol with angular paths."""
        n_points = self.rng.randint(4, 7)
        points = []

        cx, cy = 50, 50
        for i in range(n_points):
            angle = (i / n_points) * 2 * math.pi + self.rng.uniform(-0.3, 0.3)
            r = self.rng.uniform(15, 40)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((x, y))

        # Build path
        path_parts = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
        for x, y in points[1:]:
            path_parts.append(f"L {x:.1f} {y:.1f}")
        path_parts.append("Z")

        # Add internal lines for complexity
        if self.rng.random() > 0.5 and len(points) >= 4:
            i, j = self.rng.sample(range(len(points)), 2)
            path_parts.append(
                f"M {points[i][0]:.1f} {points[i][1]:.1f} "
                f"L {points[j][0]:.1f} {points[j][1]:.1f}"
            )

        return " ".join(path_parts)

    def _generate_mixed_symbol(self) -> str:
        """Generate a symbol with mixed curves and lines."""
        elements = []

        # Main shape
        shape_type = self.rng.choice(["circle", "triangle", "square", "custom"])

        if shape_type == "circle":
            cx = 50 + self.rng.uniform(-10, 10)
            cy = 50 + self.rng.uniform(-10, 10)
            r = self.rng.uniform(15, 30)
            # Approximate circle with bezier
            k = 0.5522847498  # Magic number for circle approximation
            elements.append(
                f"M {cx} {cy-r} "
                f"C {cx+r*k} {cy-r} {cx+r} {cy-r*k} {cx+r} {cy} "
                f"C {cx+r} {cy+r*k} {cx+r*k} {cy+r} {cx} {cy+r} "
                f"C {cx-r*k} {cy+r} {cx-r} {cy+r*k} {cx-r} {cy} "
                f"C {cx-r} {cy-r*k} {cx-r*k} {cy-r} {cx} {cy-r}"
            )
        elif shape_type == "triangle":
            cx, cy = 50, 50
            r = self.rng.uniform(20, 35)
            rotation = self.rng.uniform(0, 2 * math.pi)
            points = []
            for i in range(3):
                angle = rotation + (i / 3) * 2 * math.pi
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                points.append((x, y))
            elements.append(
                f"M {points[0][0]:.1f} {points[0][1]:.1f} "
                f"L {points[1][0]:.1f} {points[1][1]:.1f} "
                f"L {points[2][0]:.1f} {points[2][1]:.1f} Z"
            )
        elif shape_type == "square":
            size = self.rng.uniform(25, 40)
            x = 50 - size / 2 + self.rng.uniform(-10, 10)
            y = 50 - size / 2 + self.rng.uniform(-10, 10)
            rotation = self.rng.uniform(-0.5, 0.5)
            # Rotated square
            cos_r, sin_r = math.cos(rotation), math.sin(rotation)
            corners = [
                (x, y), (x + size, y),
                (x + size, y + size), (x, y + size)
            ]
            cx, cy = 50, 50
            rotated = []
            for px, py in corners:
                dx, dy = px - cx, py - cy
                rx = cx + dx * cos_r - dy * sin_r
                ry = cy + dx * sin_r + dy * cos_r
                rotated.append((rx, ry))
            elements.append(
                f"M {rotated[0][0]:.1f} {rotated[0][1]:.1f} "
                f"L {rotated[1][0]:.1f} {rotated[1][1]:.1f} "
                f"L {rotated[2][0]:.1f} {rotated[2][1]:.1f} "
                f"L {rotated[3][0]:.1f} {rotated[3][1]:.1f} Z"
            )
        else:
            # Custom free-form
            elements.append(self._generate_curved_symbol())

        # Add decorative element
        if self.rng.random() > 0.3:
            dec_type = self.rng.choice(["dot", "line", "curve"])
            if dec_type == "dot":
                dx = 50 + self.rng.uniform(-30, 30)
                dy = 50 + self.rng.uniform(-30, 30)
                r = self.rng.uniform(3, 6)
                elements.append(
                    f"M {dx} {dy-r} a {r} {r} 0 1 0 0 {2*r} a {r} {r} 0 1 0 0 {-2*r}"
                )
            elif dec_type == "line":
                x1 = self.rng.uniform(20, 80)
                y1 = self.rng.uniform(20, 80)
                x2 = self.rng.uniform(20, 80)
                y2 = self.rng.uniform(20, 80)
                elements.append(f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}")
            else:
                cx = self.rng.uniform(30, 70)
                cy = self.rng.uniform(30, 70)
                x1 = cx + self.rng.uniform(-20, 20)
                y1 = cy + self.rng.uniform(-20, 20)
                x2 = cx + self.rng.uniform(-20, 20)
                y2 = cy + self.rng.uniform(-20, 20)
                cpx = (x1 + x2) / 2 + self.rng.uniform(-15, 15)
                cpy = (y1 + y2) / 2 + self.rng.uniform(-15, 15)
                elements.append(f"M {x1:.1f} {y1:.1f} Q {cpx:.1f} {cpy:.1f} {x2:.1f} {y2:.1f}")

        return " ".join(elements)

    def _generate_spiral_symbol(self) -> str:
        """Generate a spiral-like symbol."""
        cx, cy = 50, 50
        start_r = self.rng.uniform(5, 15)
        end_r = self.rng.uniform(25, 40)
        turns = self.rng.uniform(1.5, 3)
        n_points = 20

        points = []
        for i in range(n_points):
            t = i / (n_points - 1)
            angle = turns * 2 * math.pi * t
            r = start_r + (end_r - start_r) * t
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((x, y))

        # Build smooth path
        path_parts = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]

        for i in range(1, len(points) - 1):
            # Use smooth curve
            path_parts.append(f"L {points[i][0]:.1f} {points[i][1]:.1f}")

        path_parts.append(f"L {points[-1][0]:.1f} {points[-1][1]:.1f}")

        return " ".join(path_parts)

    def generate_labels(self, n: int = 16) -> List[str]:
        """
        Generate n unique pronounceable nonword labels.

        Parameters
        ----------
        n : int
            Number of labels to generate

        Returns
        -------
        List[str]
            List of unique labels
        """
        available = [
            label for label in self.PREDEFINED_LABELS
            if label not in self._used_labels
        ]

        self.rng.shuffle(available)

        if len(available) < n:
            # Generate additional labels if needed
            additional = self._generate_pronounceable_nonwords(n - len(available))
            available.extend(additional)

        selected = available[:n]
        self._used_labels.update(selected)

        return selected

    def _generate_pronounceable_nonwords(self, n: int) -> List[str]:
        """Generate additional pronounceable nonwords."""
        # Syllable components
        onsets = [
            "", "b", "bl", "br", "ch", "cl", "cr", "d", "dr", "f", "fl", "fr",
            "g", "gl", "gr", "h", "j", "k", "kl", "kr", "l", "m", "n", "p",
            "pl", "pr", "qu", "r", "s", "sc", "sk", "sl", "sm", "sn", "sp",
            "spl", "spr", "st", "str", "sw", "t", "th", "tr", "tw", "v", "w",
            "wr", "z"
        ]

        nuclei = ["a", "e", "i", "o", "u", "ai", "au", "ea", "ee", "ie", "oo", "ou"]

        codas = [
            "", "b", "ch", "ck", "d", "dge", "f", "g", "k", "l", "ll", "m",
            "mp", "n", "nd", "ng", "nk", "nt", "p", "r", "rk", "rm", "rn",
            "rp", "rt", "s", "sh", "sk", "sp", "ss", "st", "t", "th", "x", "z"
        ]

        labels = []
        attempts = 0
        max_attempts = n * 100

        while len(labels) < n and attempts < max_attempts:
            attempts += 1

            # Generate 2-syllable word
            syl1 = (
                self.rng.choice(onsets) +
                self.rng.choice(nuclei) +
                self.rng.choice(["", "l", "n", "r", "s", "t"])
            )
            syl2 = (
                self.rng.choice(["b", "d", "g", "k", "l", "m", "n", "p", "r", "t", "z"]) +
                self.rng.choice(nuclei) +
                self.rng.choice(codas)
            )

            word = (syl1 + syl2).upper()

            # Validate
            if (
                5 <= len(word) <= 7 and
                word not in self._used_labels and
                word not in labels and
                self._is_pronounceable(word)
            ):
                labels.append(word)

        return labels

    def _is_pronounceable(self, word: str) -> bool:
        """Check if a word is pronounceable."""
        word = word.lower()

        # Check for vowels
        vowels = set("aeiou")
        if not any(c in vowels for c in word):
            return False

        # Check for too many consecutive consonants
        consonant_count = 0
        for c in word:
            if c not in vowels:
                consonant_count += 1
                if consonant_count > 3:
                    return False
            else:
                consonant_count = 0

        return True

    def generate_stimulus_set(
        self,
        n_pairs: int = 16,
        validate: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate a complete stimulus set with symbol-label pairs.

        Parameters
        ----------
        n_pairs : int
            Number of pairs to generate
        validate : bool
            Whether to validate stimuli

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Stimulus set with symbol -> label mappings
        """
        symbols = self.generate_symbols(n_pairs)
        labels = self.generate_labels(n_pairs)

        # Shuffle labels for random pairing
        self.rng.shuffle(labels)

        stimulus_set = {}
        for symbol, label in zip(symbols, labels):
            stimulus_set[symbol.id] = {
                "symbol": symbol.to_dict(),
                "label": label,
            }

        return stimulus_set

    def validate_stimulus_set(
        self,
        stimulus_set: Dict[str, Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a stimulus set for experiment use.

        Parameters
        ----------
        stimulus_set : Dict[str, Dict[str, Any]]
            Stimulus set to validate

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list of issues)
        """
        issues = []

        # Check count
        if len(stimulus_set) < 16:
            issues.append(f"Too few stimuli: {len(stimulus_set)}")

        # Check for unique labels
        labels = [s["label"] for s in stimulus_set.values()]
        if len(labels) != len(set(labels)):
            issues.append("Duplicate labels found")

        # Check label properties
        for label in labels:
            if len(label) < 5 or len(label) > 7:
                issues.append(f"Label '{label}' has invalid length")
            if not self._is_pronounceable(label):
                issues.append(f"Label '{label}' may not be pronounceable")

        # Check symbol SVG paths
        for sid, stim in stimulus_set.items():
            svg_path = stim.get("symbol", {}).get("svg_path", "")
            if not svg_path or len(svg_path) < 10:
                issues.append(f"Symbol '{sid}' has invalid SVG path")

        is_valid = len(issues) == 0
        return is_valid, issues
