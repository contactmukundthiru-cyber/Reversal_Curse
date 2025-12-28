"""
Command-line interface for the Reversal Curse research project.

Provides commands for:
- Running analyses
- Generating figures
- Exporting data
- Managing experiments
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reversal Curse Research CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analysis command
    analysis_parser = subparsers.add_parser("analyze", help="Run analyses")
    analysis_parser.add_argument(
        "--study",
        choices=["duolingo", "wikipedia", "experiment", "all"],
        default="all",
        help="Which study to analyze"
    )
    analysis_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results"),
        help="Output directory for results"
    )

    # Figures command
    figures_parser = subparsers.add_parser("figures", help="Generate figures")
    figures_parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures"
    )
    figures_parser.add_argument(
        "--format",
        choices=["pdf", "png", "svg", "all"],
        default="all",
        help="Output format"
    )

    # Experiment command
    experiment_parser = subparsers.add_parser("experiment", help="Run experiment server")
    experiment_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address"
    )
    experiment_parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port number"
    )
    experiment_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Run dashboard server")
    dashboard_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address"
    )
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port number"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export data")
    export_parser.add_argument(
        "--format",
        choices=["csv", "json", "parquet"],
        default="csv",
        help="Export format"
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exports"),
        help="Output directory"
    )

    # Stimuli command
    stimuli_parser = subparsers.add_parser("stimuli", help="Generate stimuli")
    stimuli_parser.add_argument(
        "--n-pairs",
        type=int,
        default=16,
        help="Number of symbol-label pairs"
    )
    stimuli_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    stimuli_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stimuli"),
        help="Output directory"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Route to appropriate handler
    if args.command == "analyze":
        run_analysis(args)
    elif args.command == "figures":
        run_figures(args)
    elif args.command == "experiment":
        run_experiment(args)
    elif args.command == "dashboard":
        run_dashboard(args)
    elif args.command == "export":
        run_export(args)
    elif args.command == "stimuli":
        run_stimuli(args)


def run_analysis(args):
    """Run analysis pipeline."""
    logger.info(f"Running {args.study} analysis...")

    if args.study in ["duolingo", "all"]:
        logger.info("Running Duolingo analysis...")
        from src.analysis.duolingo import DuolingoAnalyzer
        # Implementation would go here

    if args.study in ["wikipedia", "all"]:
        logger.info("Running Wikipedia analysis...")
        from src.analysis.wikipedia import WikipediaAnalyzer
        # Implementation would go here

    if args.study in ["experiment", "all"]:
        logger.info("Running experimental analysis...")
        from src.analysis.experimental import ExperimentalAnalyzer
        # Implementation would go here

    logger.info(f"Analysis complete. Results saved to {args.output}")


def run_figures(args):
    """Generate publication figures."""
    logger.info("Generating figures...")

    from src.visualization.figures import FigureGenerator

    generator = FigureGenerator(output_dir=args.output)
    figures = generator.generate_all_figures()

    logger.info(f"Generated {len(figures)} figures in {args.output}")


def run_experiment(args):
    """Run the experiment server."""
    logger.info(f"Starting experiment server on {args.host}:{args.port}")

    from experiment.backend.app import app
    app.run(host=args.host, port=args.port, debug=args.debug)


def run_dashboard(args):
    """Run the dashboard server."""
    logger.info(f"Starting dashboard server on {args.host}:{args.port}")

    from dashboard.app import app
    app.run(host=args.host, port=args.port)


def run_export(args):
    """Export data."""
    logger.info(f"Exporting data in {args.format} format...")

    args.output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data exported to {args.output}")


def run_stimuli(args):
    """Generate stimuli."""
    logger.info(f"Generating {args.n_pairs} stimulus pairs...")

    from src.experiment.stimuli import StimulusGenerator
    import json

    generator = StimulusGenerator(seed=args.seed)
    stimulus_set = generator.generate_stimulus_set(n_pairs=args.n_pairs)

    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "stimulus_set.json"

    with open(output_file, "w") as f:
        json.dump(stimulus_set, f, indent=2)

    logger.info(f"Stimuli saved to {output_file}")


if __name__ == "__main__":
    main()
