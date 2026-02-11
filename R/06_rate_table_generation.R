# =============================================================================
# 06 - Rate Table Generation
# French Motor Insurance - Generate premium rate relativities from GLM
# Reference: insurancerating package, actuarial best practices
# =============================================================================

library(dplyr)

# -----------------------------------------------------------------------------
# Extract relativities from GLM coefficients
# -----------------------------------------------------------------------------

extract_glm_relativities <- function(model) {
  coef_table <- as.data.frame(summary(model)$coefficients)
  coef_table$Variable <- rownames(coef_table)
  coef_table$Relativity <- exp(coef_table$Estimate)
  coef_table
}

# -----------------------------------------------------------------------------
# Generate rate table by rating factor
# -----------------------------------------------------------------------------

generate_rate_table <- function(model, data, base_level = NULL) {
  coef_df <- extract_glm_relativities(model)

  # Extract main effects (exclude interactions and offset)
  main_effects <- coef_df[!grepl(":", coef_df$Variable) & coef_df$Variable != "(Intercept)", ]

  # Build rate table by factor
  rate_tables <- list()

  # DrivAge: use coefficient for continuous variable
  if ("DrivAge" %in% main_effects$Variable) {
    age_coef <- main_effects$Estimate[main_effects$Variable == "DrivAge"]
    rate_tables$DrivAge <- data.frame(
      Factor = "DrivAge",
      Level = "per_year",
      Coefficient = age_coef,
      Relativity = exp(age_coef),
      Description = "Multiplicative effect per year of age"
    )
  }

  # Categorical factors: extract base and levels
  cat_factors <- c("Area", "VehGas", "VehBrand", "Region")
  for (fac in cat_factors) {
    fac_coefs <- main_effects[grepl(paste0("^", fac), main_effects$Variable), ]
    if (nrow(fac_coefs) > 0) {
      fac_coefs$Level <- gsub(paste0("^", fac), "", fac_coefs$Variable)
      fac_coefs$Factor <- fac
      rate_tables[[fac]] <- fac_coefs %>%
        select(Factor, Level, Estimate, Relativity = Relativity)
    }
  }

  # Continuous variables
  cont_vars <- c("VehAge", "VehPower", "BonusMalus", "LogDensity")
  for (var in cont_vars) {
    if (var %in% main_effects$Variable) {
      coef_val <- main_effects$Estimate[main_effects$Variable == var]
      rate_tables[[var]] <- data.frame(
        Factor = var,
        Level = "per_unit",
        Coefficient = coef_val,
        Relativity = exp(coef_val),
        Description = paste("Multiplicative effect per unit increase in", var)
      )
    }
  }

  rate_tables
}

# -----------------------------------------------------------------------------
# Create combined rate table for manual development
# -----------------------------------------------------------------------------

create_rating_manual <- function(model, output_path = "output/tables/rate_table.csv") {
  coef_df <- extract_glm_relativities(model)
  coef_df <- coef_df[order(coef_df$Variable), ]
  out_df <- data.frame(
    Variable = coef_df$Variable,
    Coefficient = coef_df$Estimate,
    Relativity = coef_df$Relativity,
    StdError = coef_df[["Std. Error"]],
    PValue = coef_df[["Pr(>|z|)"]],
    stringsAsFactors = FALSE
  )
  coef_df <- out_df

  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  write.csv(coef_df, output_path, row.names = FALSE)
  message("Rate table saved to ", output_path)
  invisible(coef_df)
}

# -----------------------------------------------------------------------------
# Main rate table pipeline
# -----------------------------------------------------------------------------

run_rate_table_generation <- function(model_path = "output/models/glm_best_model.rds",
                                     output_dir = "output") {
  model <- readRDS(model_path)
  data <- readRDS("data/processed/freq_data_clean.rds")

  dir.create(file.path(output_dir, "tables"), recursive = TRUE, showWarnings = FALSE)

  # Full coefficient table
  create_rating_manual(model, file.path(output_dir, "tables", "rate_table.csv"))

  # Factor-level relativities
  rate_tables <- generate_rate_table(model, data)

  # Save factor relativities
  for (fac_name in names(rate_tables)) {
    write.csv(rate_tables[[fac_name]],
      file.path(output_dir, "tables", paste0("relativity_", fac_name, ".csv")),
      row.names = FALSE
    )
  }

  cat("\nRate table generation complete!\n")
  cat("Files saved to", file.path(output_dir, "tables"), "\n")
  invisible(rate_tables)
}
