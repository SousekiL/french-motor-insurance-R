# =============================================================================
# 04 - XGBoost Modeling
# French Motor Insurance - Gradient boosting for claim frequency
# Reference: DAV GitHub Repository, Chen & Guestrin (2016)
# =============================================================================

library(xgboost)
library(ggplot2)
library(gridExtra)
library(dplyr)

# -----------------------------------------------------------------------------
# 4.1 Feature Preparation
# -----------------------------------------------------------------------------

prepare_xgb_features <- function(data) {
  model.matrix(
    ~ DrivAge + VehAge + VehPower + BonusMalus + LogDensity +
      IsYoungDriver + IsOldDriver + IsDiesel + IsHighBonus +
      Area + VehGas + VehBrand + Region +
      DrivAgeGroup + VehAgeGroup + BonusMalusGroup + VehPowerGroup - 1,
    data = data
  )
}

# -----------------------------------------------------------------------------
# 4.2 Evaluation
# -----------------------------------------------------------------------------

evaluate_xgb <- function(actual, predicted, exposure, dataset_name) {
  poisson_dev <- 2 * sum(
    ifelse(actual > 0, actual * log(actual / predicted), 0) -
      (actual - predicted)
  )
  mean_dev <- poisson_dev / sum(exposure)
  total_actual <- sum(actual)
  total_predicted <- sum(predicted)
  freq_actual <- sum(actual) / sum(exposure)
  freq_predicted <- sum(predicted) / sum(exposure)

  cat("\n=== XGBOOST EVALUATION:", dataset_name, "===\n")
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
# 4.3 Partial Dependence Plot
# -----------------------------------------------------------------------------

create_pdp <- function(model, data, feature_name, feature_values = NULL,
                       prepare_fn = prepare_xgb_features) {
  if (is.null(feature_values)) {
    feature_values <- quantile(data[[feature_name]], probs = seq(0, 1, 0.05), na.rm = TRUE)
  }

  pdp_data <- data.frame()
  for (val in unique(feature_values)) {
    temp_data <- data
    temp_data[[feature_name]] <- val
    temp_matrix <- prepare_fn(temp_data)
    temp_dmatrix <- xgb.DMatrix(data = temp_matrix)
    pred <- predict(model, temp_dmatrix)
    pdp_data <- rbind(pdp_data, data.frame(
      feature_value = val,
      prediction = mean(pred)
    ))
  }
  pdp_data
}

# -----------------------------------------------------------------------------
# Main XGBoost pipeline
# -----------------------------------------------------------------------------

run_xgboost_modeling <- function(data_path = "data/processed/freq_data_clean.rds",
                                glm_metrics_path = "output/models/glm_metrics.rds",
                                output_dir = "output") {
  freq_data_clean <- readRDS(data_path)

  # Use same train-test split as GLM
  glm_metrics <- readRDS(glm_metrics_path)
  train_indices <- glm_metrics$train_indices
  train_data <- freq_data_clean[train_indices, ]
  test_data <- freq_data_clean[-train_indices, ]

  dir.create(file.path(output_dir, "models"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(output_dir, "figures"), recursive = TRUE, showWarnings = FALSE)

  cat("=== PREPARING FEATURES FOR XGBOOST ===\n")
  X_train <- prepare_xgb_features(train_data)
  X_test <- prepare_xgb_features(test_data)
  y_train <- train_data$ClaimNb
  y_test <- test_data$ClaimNb
  w_train <- train_data$Exposure
  w_test <- test_data$Exposure

  dtrain <- xgb.DMatrix(data = X_train, label = y_train, weight = w_train)
  dtest <- xgb.DMatrix(data = X_test, label = y_test, weight = w_test)

  cat("Training features shape:", dim(X_train), "\n")
  cat("Number of features:", ncol(X_train), "\n")

  # Hyperparameter tuning (simplified - use default good params)
  cat("\n=== TRAINING XGBOOST MODEL ===\n")
  params <- list(
    objective = "count:poisson",
    eval_metric = "poisson-nloglik",
    max_depth = 5,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    gamma = 0.1
  )

  watchlist <- list(train = dtrain, test = dtest)
  xgb_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 500,
    watchlist = watchlist,
    early_stopping_rounds = 20,
    print_every_n = 50,
    verbose = 1
  )

  cat("\nBest iteration:", xgb_model$best_iteration, "\n")

  # Predictions
  train_data$pred_xgb <- predict(xgb_model, dtrain)
  test_data$pred_xgb <- predict(xgb_model, dtest)

  train_eval <- evaluate_xgb(train_data$ClaimNb, train_data$pred_xgb, train_data$Exposure, "TRAINING SET")
  test_eval <- evaluate_xgb(test_data$ClaimNb, test_data$pred_xgb, test_data$Exposure, "TEST SET")

  # Feature importance
  importance_matrix <- xgb.importance(feature_names = colnames(X_train), model = xgb_model)

  p_importance <- ggplot(head(importance_matrix, 20), aes(x = reorder(Feature, Gain), y = Gain)) +
    geom_col(fill = "#06A77D", alpha = 0.8) +
    coord_flip() +
    labs(
      title = "XGBoost Feature Importance (Top 20)",
      subtitle = "Based on total gain (improvement in accuracy)",
      x = "Feature", y = "Importance (Gain)"
    ) +
    theme_minimal()
  ggsave(file.path(output_dir, "figures", "11_xgboost_importance.png"), p_importance, width = 10, height = 8, dpi = 300)

  # Partial dependence plots
  pdp_age <- create_pdp(xgb_model, train_data, "DrivAge")
  pdp_power <- create_pdp(xgb_model, train_data, "VehPower")
  pdp_bonus <- create_pdp(xgb_model, train_data, "BonusMalus")

  p_pdp1 <- ggplot(pdp_age, aes(x = feature_value, y = prediction)) +
    geom_line(color = "#2E86AB", linewidth = 1.5) +
    labs(title = "Partial Dependence: Driver Age", x = "Driver Age", y = "Predicted Claims")

  p_pdp2 <- ggplot(pdp_power, aes(x = feature_value, y = prediction)) +
    geom_line(color = "#A23B72", linewidth = 1.5) +
    labs(title = "Partial Dependence: Vehicle Power", x = "Vehicle Power", y = "Predicted Claims")

  p_pdp3 <- ggplot(pdp_bonus, aes(x = feature_value, y = prediction)) +
    geom_line(color = "#D62246", linewidth = 1.5) +
    labs(title = "Partial Dependence: Bonus-Malus", x = "Bonus-Malus", y = "Predicted Claims")

  g <- gridExtra::arrangeGrob(p_pdp1, p_pdp2, p_pdp3, ncol = 3)
  ggsave(file.path(output_dir, "figures", "12_xgboost_pdp.png"), plot = g, width = 15, height = 5, dpi = 300)

  # Save
  xgb.save(xgb_model, file.path(output_dir, "models", "xgboost_model.model"))
  saveRDS(xgb_model, file.path(output_dir, "models", "xgboost_model.rds"))

  saveRDS(list(
    train = train_data %>% select(IDpol, ClaimNb, Exposure, pred_xgb),
    test = test_data %>% select(IDpol, ClaimNb, Exposure, pred_xgb)
  ), file.path(output_dir, "models", "xgboost_predictions.rds"))

  xgb_metrics <- list(
    train = train_eval,
    test = test_eval,
    importance = importance_matrix,
    feature_names = colnames(X_train)
  )
  saveRDS(xgb_metrics, file.path(output_dir, "models", "xgboost_metrics.rds"))

  cat("\nXGBoost modeling complete! Model saved.\n")
  list(model = xgb_model, train = train_data, test = test_data, metrics = xgb_metrics)
}
