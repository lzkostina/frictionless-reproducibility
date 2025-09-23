"""
Reproduce results using safe precomputed artifacts (no raw data required).

This script:
1. Sets an environment variable to tell the analysis to use artifacts/
2. Runs the statistical analysis pipeline
3. Saves results to results/tables/ (and figures if implemented)
"""

import os

os.environ["USE_ARTIFACTS"] = "1"

from src.analysis import run_stats


def main():
    print("=== Running analysis from artifacts (safe mode) ===")
    run_stats.run()


if __name__ == "__main__":
    main()
