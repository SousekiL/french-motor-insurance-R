# =============================================================================
# Evaluation Metrics
# French Motor Insurance - Model evaluation functions
# =============================================================================

#' Calculate Root Mean Squared Error
#'
#' @param actual Numeric. Actual values
#' @param predicted Numeric. Predicted values
#' @return Numeric RMSE
calc_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

#' Calculate Mean Absolute Error
#'
#' @param actual Numeric. Actual values
#' @param predicted Numeric. Predicted values
#' @return Numeric MAE
calc_mae <- function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}

#' Calculate Gini coefficient (for insurance pricing)
#'
#' @param actual Numeric. Actual values
#' @param predicted Numeric. Predicted values
#' @param exposure Numeric. Exposure weights (optional)
#' @return Numeric Gini
calc_gini <- function(actual, predicted, exposure = NULL) {
  # TODO: Implement Gini for model lift
  if (is.null(exposure)) exposure <- rep(1, length(actual))
  # Simplified version - full implementation depends on use case
  invisible(NULL)
}

#' Calculate deviance for Poisson GLM
#'
#' @param actual Numeric. Actual claim counts
#' @param predicted Numeric. Predicted claim counts
#' @return Numeric deviance
calc_poisson_deviance <- function(actual, predicted) {
  2 * sum(ifelse(actual == 0, predicted, actual * log(actual / predicted) - (actual - predicted)), na.rm = TRUE)
}
