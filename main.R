# =============================================================================
# French Motor Insurance - Main Pipeline
# Run from project root: setwd("path/to/french-motor-insurance-R"); source("main.R")
# =============================================================================

options(scipen = 999)
set.seed(42)

# Source helper functions
source("functions/helper_functions.R")
source("functions/evaluation_metrics.R")

# -----------------------------------------------------------------------------
# Step 1: Data Loading
# -----------------------------------------------------------------------------
cat("\n========== STEP 1: DATA LOADING ==========\n")
source("R/01_data_loading.R")
freq_data_clean <- run_data_loading()

# -----------------------------------------------------------------------------
# Step 2: Exploratory Analysis
# -----------------------------------------------------------------------------
cat("\n========== STEP 2: EXPLORATORY ANALYSIS ==========\n")
source("R/02_exploratory_analysis.R")
run_eda(data = freq_data_clean)

# -----------------------------------------------------------------------------
# Step 3: GLM Modeling
# -----------------------------------------------------------------------------
cat("\n========== STEP 3: GLM MODELING ==========\n")
source("R/03_glm_modeling.R")
glm_result <- run_glm_modeling()

# -----------------------------------------------------------------------------
# Step 4: XGBoost Modeling
# -----------------------------------------------------------------------------
cat("\n========== STEP 4: XGBOOST MODELING ==========\n")
source("R/04_xgboost_modeling.R")
xgb_result <- run_xgboost_modeling()

# -----------------------------------------------------------------------------
# Step 5: Model Comparison
# -----------------------------------------------------------------------------
cat("\n========== STEP 5: MODEL COMPARISON ==========\n")
source("R/05_model_comparison.R")
comparison <- run_model_comparison()

# -----------------------------------------------------------------------------
# Step 6: Rate Table Generation
# -----------------------------------------------------------------------------
cat("\n========== STEP 6: RATE TABLE GENERATION ==========\n")
source("R/06_rate_table_generation.R")
rate_tables <- run_rate_table_generation()

cat("\n========================================\n")
cat("Pipeline completed successfully!\n")
cat("========================================\n")
