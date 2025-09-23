## Frictionless Reproducibility Project

### Overview
This repository transforms a previous, unstructured analysis into a fully reproducible and maintainable project.  
It implements automated pipelines, version control, environment management, and testing so that all results can be regenerated with a single command.  

The underlying research focuses on brain connectivity analysis, specifically evaluating how different network summary statistics capture the structure of brain networks.  

By running one command, you can:
- Fetch and preprocess the raw data  
- Run the complete analysis pipeline  
- Generate all tables, figures, and the final report  

All outputs are stored in the `results/` directory.

### Repository Structure
```
frictionless-reproducibility/
├── data/
│ ├── connectomes/ # large immutable brain image data (not tracked by git)
│ ├── raw/ # immutable source data (read-only; not tracked by git)
│ └── processed/ # cleaned data derived from raw and connectomes (not tracked by git)
├── notebooks/ # exploratory Jupyter notebooks
├── src/ # source code for analysis
│ ├── evaluation/ # evaluation scripts and metrics
│ ├── features/ # feature engineering code
│ ├── network/ # network analysis methods
│ ├── pipeline/ # code for preprocessing, cleaning, feature creation
│ ├── utils/ # helper functions and utilities
│ └── analysis/ # statistical models, plots, summary analysis
├── artifacts/ # intermediate outputs 
├── results/ # findings
│ ├── tables/ # summary tables
│ ├── figures/ # plots and visualizations
│ └── report/ # generated report
├── tests/ # unit and integration tests (pytest)
├── docs/ # additional notes or documentation
├── requirements.txt # Python dependencies (or environment.yml)
├── make_artifacts.py # generate artifacts for safe mode analysis
├── run_analysis.py # single entry point for end-to-end run
├── run_from_artifacts.py # save mode analysis
├── README.md # project overview and instructions
└── .gitignore # specifies files to ignore in version control
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/lzkostina/frictionless-reproducibility
cd frictionless-reproducibility
```

2. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage
#### Running with Full Data
If you have access to the raw dataset (not included due to size/sensitivity):
```bash
python run_analysis.py
```

This will:
* Load raw data (from data/raw/)
* Process and validate data → data/processed/
* Run pipeline and save models → artifacts/
* Generate final results (tables/figures/report) → results/

#### Reproduce from Artifacts (safe mode, no raw data required)
If you do not have access to the raw data, you can still reproduce the analysis using precomputed artifacts included in this repository.
```bash
python run_from_artifacts.py
```

This will:
* Load safe reduced files from artifacts/version_B/
* Skip all connectome-related steps
* Run the statistical analysis using only demographic and behavioral features
* Save results into:
* Tables: results/tables/


#### Testing

This project includes basic tests to ensure pipeline correctness.

To run all tests:
```bash
pytest tests/
```

