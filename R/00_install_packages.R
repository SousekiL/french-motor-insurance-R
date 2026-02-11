# =============================================================================
# French Motor Insurance Pricing - Package Installation
# Project: GLM vs Machine Learning Comparison for Actuarial Pricing
# Data Source: CASdatasets package (freMTPL2freq, freMTPL2sev)
# References: Wüthrich & Buser (2018), DAV GitHub Repository
# =============================================================================

# Install required packages (run once)
install.packages(c(
  # Data manipulation
  "dplyr",
  "tidyr",
  "data.table",

  # Visualization
  "ggplot2",
  "gridExtra",
  "corrplot",
  "RColorBrewer",

  # Modeling - GLM
  "statmod",
  "MASS",
  "mgcv",

  # Modeling - Machine Learning
  "xgboost",
  "randomForest",
  "gbm",

  # Actuarial packages
  "insuranceData",
  "CASdatasets",
  "actuar",
  "ChainLadder",

  # Model evaluation
  "caret",
  "pROC",
  "MLmetrics",

  # Utilities
  "here",
  "glue",
  "scales"
))

# Install CASdatasets from R-Forge (special repository)
# This package contains the freMTPL2 datasets
install.packages("CASdatasets", repos = "http://cas.uqam.ca/pub/R/", type = "source")

# Load core libraries
library(dplyr)
library(ggplot2)
library(CASdatasets)

# Set global options
options(scipen = 999)
set.seed(42)

# Verify installation
cat("All packages installed successfully!\n")
cat("R version:", R.version.string, "\n")
