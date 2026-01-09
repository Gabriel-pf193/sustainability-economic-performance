# Can Sustainability Indicators Predict Economic Performance ?
**Category:** Regression & Model Comparison Study

## Research Question
Does a relationship exist between _Environmental, Social, and Governance (ESG)_ indicators and the economic growth of countries, and can these indicators be used to predict it ?

The analysis combines an econometric fixed-effects regression with supervised machine learning models to examine explanatory and predictive performance of ESG indicators.

## Setup

### Create Environment
After cloning the GitHub repository to a local machine, create the conda environment using:
```bash
conda env create -f environment.yml
```
Once the environment is created, activate it with:
```bash
conda activate esg-eco-project
```

## Usage
The project is organized as a pipeline controlled by a single entry point: ```main.py```

Running this script executes sequentially all steps of the analysis.

Expected output:

- **Console output summarizing:**
  - Dataset sizes and shapes
  - Main regression results
  - Machine learning models performance results

- **Generated files:**
  - Processed datasets (`.csv`)
  - Descriptive sample tables and plots (`.pdf` or `.html`)
  - Regression table (`.tex`)
  - Model comparison tables and plots (`.pdf`)

## Project Structure
The project repository is organized as follows:

```bash
sustainability-economic-performance/
│
├── main.py                          # Main entry point
├── environment.yml                  # Conda environment for reproducibility
├── report.pdf                       # Project report
├── AI_USAGE.md                      # AI usage declaration
├── PROPOSAL.md                      # Initial project proposal
├── README.md                        # This document
│
├── data/
│   ├── raw/                         # Raw data (World Bank datasets)
│   └── processed/                   # Cleaned and processed datasets (.csv)
│
├── src/
│   ├── PART_1_data_preparation.py   # Data loading, cleaning, merging
│   ├── PART_2_country_selection.py  # Country selection + descriptive outputs
│   ├── PART_3_FE_regression.py      # Fixed-effects regression
│   └── PART_4_ML_models.py          # Machine learning models and evaluation
│
├── results/
│   ├── country_selection/           # HTML tables and PDF descriptive plots
│   ├── regression/                  # Regression outputs (.tex)
│   └── machine_learning/            # ML plots, learning curves, comparisons (.pdf)
│
└── notebooks/                       # Jupyter notebooks (exploration only, not required for reproduction)
```

## Results

### Fixed-effects regression (country and year FE)

- Number of observations: **1,117**
- R²: **0.516**
- Adjusted R²: **0.482**
- Standard errors: **clustered at country level**

Estimated coefficients for ESG indices:
- Environmental index (ENV): **−0.6877**
- Social index (SOC): **−0.0427**
- Governance index (GOV): **+0.3860**

### Machine learning model performance

| Model               | CV R² | CV RMSE | Test R² | Test RMSE |
|---------------------|------:|--------:|--------:|----------:|
| Linear Regression   | 0.022 | 3.464   | 0.143   | 3.137     |
| Random Forest       | 0.166 | 3.200   | 0.327   | 2.781     |
| Gradient Boosting   | 0.103 | 3.318   | 0.358   | 2.715     |

### Generated outputs

- Processed datasets
- Regression table
- Learning curves, feature importance plots and descriptive figures

## Requirements

Python 3.11 or newer

### Main Python packages
- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- openpyxl