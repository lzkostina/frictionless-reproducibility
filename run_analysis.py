"""
Full pipeline runner: from raw data to results.

Usage:
    python run_analysis.py
"""

from src.pipeline import preprocess
from src.analysis import run_stats


def main():
    print("=== Step 1: Preprocessing raw data ===")
    preprocess.run()

    print("=== Step 2: Running statistical analysis ===")
    run_stats.run()

    print("✅ Full pipeline complete. Results saved to results/")


if __name__ == "__main__":
    main()
