# French Motor Insurance Pricing

An R project for motor insurance claim frequency modeling, comparing Generalized Linear Models (GLM) with XGBoost machine learning approaches. Uses the freMTPL2 dataset from CASdatasets.

## Overview

This project implements a complete actuarial pricing workflow:

1. **Data loading** – Load and preprocess French motor insurance data (freMTPL2)
2. **Exploratory analysis** – Risk factor analysis, correlation, segment profiling
3. **GLM modeling** – Poisson and Negative Binomial regression with interactions
4. **XGBoost modeling** – Gradient boosting with Poisson objective
5. **Model comparison** – Poisson deviance, RMSE, MAE
6. **Rate table generation** – Extract premium relativities from GLM coefficients

## References

- **DAV Official Repository**: [DeutscheAktuarvereinigung/claim_frequency](https://github.com/DeutscheAktuarvereinigung/claim_frequency)
- **insurancerating Package**: [MHaringa/insurancerating](https://github.com/MHaringa/insurancerating)
- **Tutorial**: [lorentzenchr/Tutorial_freMTPL2](https://github.com/lorentzenchr/Tutorial_freMTPL2)
- **Wüthrich & Buser (2018)**: Statistical foundations of actuarial learning

## Project Structure

```
french-motor-insurance-R/
├── data/
│   ├── raw/
│   └── processed/          # freq_data_clean.rds
├── R/
│   ├── 00_install_packages.R
│   ├── 01_data_loading.R
│   ├── 02_exploratory_analysis.R
│   ├── 03_glm_modeling.R
│   ├── 04_xgboost_modeling.R
│   ├── 05_model_comparison.R
│   └── 06_rate_table_generation.R
├── output/
│   ├── figures/            # EDA plots, model diagnostics
│   ├── tables/             # Model comparison, rate tables
│   └── models/             # Saved GLM and XGBoost models
├── functions/
│   ├── helper_functions.R
│   └── evaluation_metrics.R
├── main.R
└── README.md
```

## Setup

### 1. Install packages (run once)

```r
source("R/00_install_packages.R")
```

For CASdatasets (freMTPL2 data):

```r
install.packages("CASdatasets", repos = "http://cas.uqam.ca/pub/R/", type = "source")
```

### 2. Run full pipeline

```r
setwd("path/to/french-motor-insurance-R")
source("main.R")
```

### 3. Run individual steps

```r
# Data loading only
source("R/01_data_loading.R")
run_data_loading()

# Exploratory data analysis
source("R/02_exploratory_analysis.R")
run_eda()

# GLM modeling
source("R/03_glm_modeling.R")
run_glm_modeling()

# XGBoost modeling
source("R/04_xgboost_modeling.R")
run_xgboost_modeling()

# Model comparison
source("R/05_model_comparison.R")
run_model_comparison()

# Rate table generation
source("R/06_rate_table_generation.R")
run_rate_table_generation()
```

## Dependencies

- dplyr, tidyr, data.table
- ggplot2, gridExtra, corrplot, RColorBrewer, scales
- statmod, MASS, mgcv
- xgboost
- CASdatasets (R-Forge)

## Outputs

| File | Description |
|------|-------------|
| `output/figures/01-07_*.png` | EDA visualizations |
| `output/figures/08-10_*.png` | GLM diagnostics |
| `output/figures/11-12_*.png` | XGBoost importance and PDPs |
| `output/figures/13_*.png` | Model comparison |
| `output/models/glm_best_model.rds` | Fitted GLM |
| `output/models/xgboost_model.rds` | Fitted XGBoost |
| `output/tables/model_comparison.csv` | GLM vs XGBoost metrics |
| `output/tables/rate_table.csv` | Coefficient relativities |

## License

MIT
