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
├── artifacts/ # intermediate outputs (models, stats)
├── results/ # findings
│ ├── tables/ # summary tables
│ ├── figures/ # plots and visualizations
│ └── report/ # generated report
├── tests/ # unit and integration tests (pytest)
├── docs/ # additional notes or documentation
├── requirements.txt # Python dependencies (or environment.yml)
├── run_analysis.py # single entry point for end-to-end run
├── README.md # project overview and instructions
└── .gitignore # specifies files to ignore in version control
```

### Installation
1. Clone the repository:


    $ git clone https://github.com/lzkostina/frictionless-reproducibility

    $ cd frictionless-reproducibility

2. Create environment and install dependencies:


    $ python3 -m venv .venv

    $ source .venv/bin/activate

    $ pip install -r requirements.txt

### Usage
   
    $ python3 run_analysis.py


This will:
* Load raw data (from data/raw/)
* Process and validate data → data/processed/
* Run pipeline and save models → artifacts/
* Generate final results (tables/figures/report) → results/