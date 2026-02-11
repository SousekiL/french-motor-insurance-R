# =============================================================================
# 05 - Model Comparison
# French Motor Insurance - Compare GLM vs XGBoost performance
# Reference: DAV GitHub, insurance_premium_models
# =============================================================================

library(dplyr)
library(ggplot2)

source("functions/evaluation_metrics.R")

# -----------------------------------------------------------------------------
# Compare models
# -----------------------------------------------------------------------------

compare_models <- function(glm_pred, xgb_pred, actual, exposure) {
  poisson_dev_glm <- 2 * sum(
    ifelse(actual > 0, actual * log(actual / glm_pred), 0) - (actual - glm_pred)
  )
  poisson_dev_xgb <- 2 * sum(
    ifelse(actual > 0, actual * log(actual / xgb_pred), 0) - (actual - xgb_pred)
  )

  total_exposure <- sum(exposure)
  total_actual <- sum(actual)

  comparison <- data.frame(
    Model = c("GLM", "XGBoost"),
    Poisson_Deviance = c(poisson_dev_glm, poisson_dev_xgb),
    Mean_Deviance = c(poisson_dev_glm / total_exposure, poisson_dev_xgb / total_exposure),
    Predicted_Total_GLM = c(sum(glm_pred), NA),
    Predicted_Total_XGB = c(NA, sum(xgb_pred)),
    Actual_Total = total_actual,
    AE_GLM = total_actual / sum(glm_pred),
    AE_XGB = total_actual / sum(xgb_pred),
    RMSE = c(calc_rmse(actual, glm_pred), calc_rmse(actual, xgb_pred)),
    MAE = c(calc_mae(actual, glm_pred), calc_mae(actual, xgb_pred))
  )

  comparison$Predicted_Total <- ifelse(comparison$Model == "GLM",
    comparison$Predicted_Total_GLM,
    comparison$Predicted_Total_XGB
  )
  comparison$Predicted_Total_GLM <- NULL
  comparison$Predicted_Total_XGB <- NULL
  comparison$AE <- ifelse(comparison$Model == "GLM", comparison$AE_GLM, comparison$AE_XGB)
  comparison$AE_GLM <- NULL
  comparison$AE_XGB <- NULL

  comparison
}

# -----------------------------------------------------------------------------
# Save comparison report
# -----------------------------------------------------------------------------

save_comparison_report <- function(comparison, output_path = "output/tables/model_comparison.csv") {
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  write.csv(comparison, output_path, row.names = FALSE)
  message("Comparison report saved to ", output_path)
}

# -----------------------------------------------------------------------------
# Main comparison runner
# -----------------------------------------------------------------------------

run_model_comparison <- function(output_dir = "output") {
  glm_pred <- readRDS(file.path(output_dir, "models", "glm_predictions.rds"))
  xgb_pred <- readRDS(file.path(output_dir, "models", "xgboost_predictions.rds"))
  freq_data <- readRDS("data/processed/freq_data_clean.rds")
  glm_metrics <- readRDS(file.path(output_dir, "models", "glm_metrics.rds"))
  train_idx <- glm_metrics$train_indices
  test_idx <- setdiff(seq_len(nrow(freq_data)), train_idx)

  # Test set comparison
  test_glm <- glm_pred$test
  test_xgb <- xgb_pred$test

  # Merge by IDpol to ensure alignment
  test_merged <- test_glm %>%
    inner_join(test_xgb %>% select(IDpol, pred_xgb), by = "IDpol") %>%
    rename(actual = ClaimNb, exposure = Exposure)

  comparison <- compare_models(
    test_merged$pred_glm,
    test_merged$pred_xgb,
    test_merged$actual,
    test_merged$exposure
  )

  cat("\n=== MODEL COMPARISON (Test Set) ===\n")
  print(comparison)

  # Visualization: Deviance comparison
  p <- ggplot(comparison, aes(x = Model, y = Poisson_Deviance, fill = Model)) +
    geom_col(alpha = 0.8, show.legend = FALSE) +
    geom_text(aes(label = round(Poisson_Deviance, 0)), vjust = -0.5) +
    scale_fill_manual(values = c("GLM" = "#2E86AB", "XGBoost" = "#06A77D")) +
    labs(
      title = "Model Comparison: Poisson Deviance",
      subtitle = "Lower is better",
      x = "Model",
      y = "Poisson Deviance"
    ) +
    theme_minimal()
  ggsave(file.path(output_dir, "figures", "13_model_comparison.png"), p, width = 8, height = 6, dpi = 300)

  save_comparison_report(comparison, file.path(output_dir, "tables", "model_comparison.csv"))

  cat("\nModel comparison complete!\n")
  invisible(comparison)
}
