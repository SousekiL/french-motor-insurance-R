# =============================================================================
# 03 - GLM Modeling
# French Motor Insurance - Poisson GLM for claim frequency
# Reference: Wüthrich & Buser (2018), De Jong & Heller (2008)
# =============================================================================

library(statmod)
library(MASS)
library(ggplot2)
library(gridExtra)
library(dplyr)

# -----------------------------------------------------------------------------
# 3.1 Train-Test Split
# -----------------------------------------------------------------------------

get_train_test_split <- function(data, seed = 42, train_pct = 0.8) {
  set.seed(seed)
  n <- nrow(data)
  train_idx <- sample(1:n, size = floor(train_pct * n))
  list(
    train = data[train_idx, ],
    test = data[-train_idx, ],
    train_indices = train_idx
  )
}

# -----------------------------------------------------------------------------
# 3.2 GLM Formulas
# -----------------------------------------------------------------------------

formula_glm_main <- ClaimNb ~
  DrivAge + VehAge + VehPower + VehGas + BonusMalus +
  VehBrand + Area + LogDensity + Region +
  offset(log(Exposure))

formula_glm_interactions <- ClaimNb ~
  DrivAge + VehAge + VehPower + VehGas + BonusMalus +
  VehBrand + Area + LogDensity + Region +
  DrivAge:VehPower + BonusMalus:VehPower +
  offset(log(Exposure))

# -----------------------------------------------------------------------------
# 3.3 Fit GLM
# -----------------------------------------------------------------------------

fit_glm <- function(data, formula = formula_glm_main, family = poisson(link = "log")) {
  glm(
    formula = formula,
    data = data,
    family = family,
    weights = Exposure
  )
}

# -----------------------------------------------------------------------------
# 3.4 Evaluation
# -----------------------------------------------------------------------------

evaluate_glm <- function(actual, predicted, exposure, dataset_name) {
  poisson_dev <- 2 * sum(
    ifelse(actual > 0, actual * log(actual / predicted), 0) -
      (actual - predicted)
  )
  mean_dev <- poisson_dev / sum(exposure)
  total_actual <- sum(actual)
  total_predicted <- sum(predicted)
  freq_actual <- sum(actual) / sum(exposure)
  freq_predicted <- sum(predicted) / sum(exposure)

  cat("\n=== GLM EVALUATION:", dataset_name, "===\n")
  cat("Poisson Deviance:", round(poisson_dev, 2), "\n")
  cat("Mean Deviance:", round(mean_dev, 6), "\n")
  cat("Actual total claims:", round(total_actual, 0), "\n")
  cat("Predicted total claims:", round(total_predicted, 0), "\n")
  cat("Prediction accuracy:", sprintf("%+.2f%%", (total_predicted / total_actual - 1) * 100), "\n")
  cat("Actual frequency:", round(freq_actual, 6), "\n")
  cat("Predicted frequency:", round(freq_predicted, 6), "\n")

  list(
    deviance = poisson_dev,
    mean_deviance = mean_dev,
    total_actual = total_actual,
    total_predicted = total_predicted
  )
}

# -----------------------------------------------------------------------------
# Main GLM pipeline
# -----------------------------------------------------------------------------

run_glm_modeling <- function(data_path = "data/processed/freq_data_clean.rds",
                              output_dir = "output") {
  freq_data_clean <- readRDS(data_path)
  split <- get_train_test_split(freq_data_clean)
  train_data <- split$train
  test_data <- split$test

  dir.create(file.path(output_dir, "models"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(output_dir, "figures"), recursive = TRUE, showWarnings = FALSE)

  cat("=== DATA SPLIT ===\n")
  cat("Training set size:", nrow(train_data), "\n")
  cat("Test set size:", nrow(test_data), "\n")

  # Model 1: Main effects
  cat("\n=== TRAINING GLM MODEL 1: MAIN EFFECTS ===\n")
  glm_model1 <- fit_glm(train_data, formula_glm_main)
  dispersion_param <- sum(residuals(glm_model1, type = "pearson")^2) / glm_model1$df.residual
  cat("Dispersion parameter:", round(dispersion_param, 3), "\n")

  # Model 2: With interactions
  cat("\n=== TRAINING GLM MODEL 2: WITH INTERACTIONS ===\n")
  glm_model2 <- fit_glm(train_data, formula_glm_interactions)

  # Model 3: Negative Binomial
  cat("\n=== TRAINING MODEL 3: NEGATIVE BINOMIAL ===\n")
  nb_model <- glm.nb(formula_glm_main, data = train_data, weights = Exposure, link = log)

  # Model comparison
  model_comparison <- data.frame(
    Model = c("GLM Main Effects", "GLM with Interactions", "Negative Binomial"),
    AIC = c(AIC(glm_model1), AIC(glm_model2), AIC(nb_model)),
    BIC = c(BIC(glm_model1), BIC(glm_model2), BIC(nb_model)),
    Deviance = c(deviance(glm_model1), deviance(glm_model2), deviance(nb_model))
  )
  cat("\n=== MODEL COMPARISON ===\n")
  print(model_comparison)

  best_model <- glm_model2

  # Predictions
  train_data$pred_glm <- predict(best_model, newdata = train_data, type = "response")
  test_data$pred_glm <- predict(best_model, newdata = test_data, type = "response")

  train_eval <- evaluate_glm(train_data$ClaimNb, train_data$pred_glm, train_data$Exposure, "TRAINING SET")
  test_eval <- evaluate_glm(test_data$ClaimNb, test_data$pred_glm, test_data$Exposure, "TEST SET")

  # Coefficient analysis
  coef_table <- as.data.frame(summary(best_model)$coefficients)
  coef_table$Variable <- rownames(coef_table)
  coef_table$RelativeRisk <- exp(coef_table$Estimate)
  coef_table <- coef_table[order(-abs(coef_table$Estimate)), ]

  # Coefficient plot
  sig_coefs <- coef_table[coef_table$`Pr(>|z|)` < 0.05, ]
  sig_coefs <- head(sig_coefs[order(-abs(sig_coefs$Estimate)), ], 20)

  p_coef <- ggplot(sig_coefs, aes(x = reorder(Variable, Estimate), y = Estimate)) +
    geom_col(aes(fill = Estimate > 0), alpha = 0.8, show.legend = FALSE) +
    scale_fill_manual(values = c("TRUE" = "#E46726", "FALSE" = "#6D9EC1")) +
    coord_flip() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    labs(
      title = "GLM Coefficients (Top 20 Significant Factors)",
      subtitle = "Positive = increases claim frequency, Negative = decreases",
      x = "Variable", y = "Coefficient (log scale)"
    ) +
    theme_minimal()
  ggsave(file.path(output_dir, "figures", "08_glm_coefficients.png"), p_coef, width = 10, height = 8, dpi = 300)

  # Residual diagnostics
  train_data$residuals <- residuals(best_model, type = "pearson")
  p_resid1 <- ggplot(train_data, aes(x = pred_glm, y = residuals)) +
    geom_point(alpha = 0.2, size = 0.5) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    geom_smooth(method = "loess", color = "blue", se = TRUE) +
    labs(title = "GLM Residual Plot", x = "Predicted Claims", y = "Pearson Residuals")

  p_resid2 <- ggplot(train_data, aes(sample = residuals)) +
    stat_qq(alpha = 0.3) +
    stat_qq_line(color = "red", linetype = "dashed") +
    labs(title = "Q-Q Plot of Residuals", x = "Theoretical Quantiles", y = "Sample Quantiles")

  g <- gridExtra::arrangeGrob(p_resid1, p_resid2, ncol = 2)
  ggsave(file.path(output_dir, "figures", "09_glm_residuals.png"), plot = g, width = 12, height = 5, dpi = 300)

  # A/E by age group
  age_performance <- test_data %>%
    group_by(DrivAgeGroup) %>%
    summarise(
      Policies = n(),
      ActualClaims = sum(ClaimNb),
      PredictedClaims = sum(pred_glm),
      Exposure = sum(Exposure),
      ActualFreq = ActualClaims / Exposure,
      PredictedFreq = PredictedClaims / Exposure,
      AE_Ratio = ActualClaims / PredictedClaims,
      .groups = "drop"
    )

  p_ae <- ggplot(age_performance, aes(x = DrivAgeGroup, y = AE_Ratio)) +
    geom_col(fill = "#2E86AB", alpha = 0.8) +
    geom_hline(yintercept = 1, color = "red", linetype = "dashed", linewidth = 1) +
    geom_text(aes(label = sprintf("%.3f", AE_Ratio)), vjust = -0.5) +
    labs(
      title = "GLM Actual/Expected Ratios by Age Group",
      subtitle = "A/E = 1.0 indicates perfect prediction",
      x = "Driver Age Group", y = "Actual/Expected Ratio"
    ) +
    ylim(0, max(age_performance$AE_Ratio) * 1.2)
  ggsave(file.path(output_dir, "figures", "10_glm_ae_ratios.png"), p_ae, width = 10, height = 6, dpi = 300)

  # Save outputs
  saveRDS(best_model, file.path(output_dir, "models", "glm_best_model.rds"))
  saveRDS(list(
    train = train_data %>% select(IDpol, ClaimNb, Exposure, pred_glm, residuals),
    test = test_data %>% select(IDpol, ClaimNb, Exposure, pred_glm)
  ), file.path(output_dir, "models", "glm_predictions.rds"))

  glm_metrics <- list(
    train = train_eval,
    test = test_eval,
    model_comparison = model_comparison,
    age_performance = age_performance,
    coefficients = coef_table,
    train_indices = split$train_indices
  )
  saveRDS(glm_metrics, file.path(output_dir, "models", "glm_metrics.rds"))

  cat("\nGLM modeling complete! Model saved.\n")
  list(model = best_model, train = train_data, test = test_data, metrics = glm_metrics)
}
