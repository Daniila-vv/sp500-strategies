# sp500-strategies

## Environment Setup and Script Execution

### Requirements
Before starting, ensure you have:

- Python version 3.10.12 or compatible installed.
- pip for dependency installation.

### Setting Up the Environment
1. Clone the Repository:
``` 
git clone https://01.kood.tech/git/dvorontso/sp500-strategies
```
2. Create a Virtual Environment 
```
python -m venv .venv
source .venv/bin/activate    # For macOS/Linux
.venv\Scripts\activate       # For Windows
```
3. Install Dependencies:
``` 
pip install -r requirements.txt
```

## Script Execution Order

1. Data Analysis:
```
python scripts/data_analysis.py
```
Performs initial data analysis, checks data integrity, and identifies anomalies.

2. Feature Engineering:
```
python scripts/features_engineering.py
```
Generates new features from the original data for model training.

3. Model Training and Hyperparameter Search:

```
python scripts/gridsearch.py
```
Conducts hyperparameter optimization using cross-validation and saves the best model.

4. Generate Additional Files:
```
python scripts/create_missing_files.py
```
Creates necessary intermediate files required for subsequent steps.

5. Generate Trading Signal:
``` 
python scripts/create_signal.py
```
Produces trading signals based on the trained model.

6. Run the Trading Strategy:
```
python scripts/strategy.py
```
Tests the strategy using the generated signals and saves the results.

## Viewing Results

All results are saved in the results/ directory:
* Training metrics and plots: results/cross-validation/
* Model and trading signals: results/selected-model/
* Strategy results: results/strategy/

## Report
For a detailed project analysis and conclusions, refer to results/strategy/report.md.


### Project Structure
```
sp500-strategies/
│
├── data/                       # Input data
│   
│
├── results/                    # Output results
│   ├── cross-validation/       # Training metrics
│   ├── selected-model/         # Trained model and signals
│   ├── strategy/               # Strategy results
│   
│
├── scripts/                    # Main scripts
│   ├── data_analysis.py        # Data analysis
│   ├── features_engineering.py # Feature engineering
│   ├── gridsearch.py           # Hyperparameter optimization
│   ├── create_missing_files.py # Generate intermediate files
│   ├── create_signal.py        # Signal generation
│   └── strategy.py             # Strategy implementation
│
├── README.md                   # Instructions
├── requirements.txt            # Dependencies
└── .venv/                      # Virtual environment (optional)
```
