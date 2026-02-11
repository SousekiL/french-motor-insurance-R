# =============================================================================
# 01 - Data Loading and Preparation
# French Motor Insurance - Load freMTPL2 data and perform initial cleaning
# Dataset: freMTPL2freq (frequency), freMTPL2sev (severity)
# Reference: Wüthrich & Buser (2018)
# =============================================================================

library(dplyr)
library(data.table)

# -----------------------------------------------------------------------------
# 1.1 Load Data
# -----------------------------------------------------------------------------

#' Load raw motor insurance data from CASdatasets or CSV files
load_raw_data <- function(raw_dir = "data/raw") {
  freq_path <- file.path(raw_dir, "freMTPL2freq.csv")
  sev_path <- file.path(raw_dir, "freMTPL2sev.csv")

  # Prefer CASdatasets if available
  if (requireNamespace("CASdatasets", quietly = TRUE)) {
    data("freMTPL2freq", package = "CASdatasets")
    data("freMTPL2sev", package = "CASdatasets")
    freq_data <- as.data.table(get("freMTPL2freq"))
    sev_data <- as.data.table(get("freMTPL2sev"))
    # Export to CSV for notebook and future runs without CASdatasets
    dir.create(raw_dir, showWarnings = FALSE, recursive = TRUE)
    write.csv(freq_data, freq_path, row.names = FALSE)
    write.csv(sev_data, sev_path, row.names = FALSE)
    message("Data loaded from CASdatasets and saved to ", raw_dir)
    return(list(freq = freq_data, sev = sev_data))
  }

  # Fallback: load from CSV if files exist
  if (file.exists(freq_path) && file.exists(sev_path)) {
    freq_data <- as.data.table(read.csv(freq_path, header = TRUE))
    sev_data <- as.data.table(read.csv(sev_path, header = TRUE))
    return(list(freq = freq_data, sev = sev_data))
  }

  stop(
    "Data not found. Either install CASdatasets (see R/00_install_packages.R)\n",
    "  or place freMTPL2freq.csv and freMTPL2sev.csv in ", raw_dir, "\n",
    "  Download from: https://www.kaggle.com/datasets/floser/french-motor-claims-datasets-fremtpl2freq"
  )
}

#' Preprocess frequency data for modeling
preprocess_data <- function(freq_data) {
  # Data quality: filter valid exposure
  freq_data_clean <- freq_data %>%
    filter(
      Exposure > 0,
      Exposure <= 1,
      Area != ""
    ) %>%
    mutate(
      Frequency = ClaimNb / Exposure,
      DrivAgeGroup = cut(DrivAge,
        breaks = c(17, 24, 34, 44, 54, 64, 100),
        labels = c("18-24", "25-34", "35-44", "45-54", "55-64", "65+"),
        include.lowest = TRUE
      ),
      VehAgeGroup = cut(VehAge,
        breaks = c(-1, 1, 4, 9, 100),
        labels = c("0-1", "2-4", "5-9", "10+"),
        include.lowest = TRUE
      ),
      BonusMalusGroup = cut(BonusMalus,
        breaks = c(49, 59, 79, 99, 119, 150, 350),
        labels = c("50-59", "60-79", "80-99", "100-119", "120-150", "150+"),
        include.lowest = TRUE
      ),
      DensityGroup = cut(Density,
        breaks = c(0, 100, 500, 2000, 10000, 100000),
        labels = c("Rural", "Low", "Medium", "High", "Very High"),
        include.lowest = TRUE
      ),
      LogDensity = log(Density + 1),
      VehPowerGroup = cut(VehPower,
        breaks = c(3, 5, 7, 9, 15),
        labels = c("4-5", "6-7", "8-9", "10+"),
        include.lowest = TRUE
      ),
      YoungPowerful = as.integer(DrivAge < 25 & VehPower > 9),
      IsYoungDriver = as.integer(DrivAge < 25),
      IsOldDriver = as.integer(DrivAge >= 65),
      IsDiesel = as.integer(VehGas == "Diesel"),
      IsHighBonus = as.integer(BonusMalus > 100)
    )

  factor_vars <- c(
    "Area", "VehBrand", "VehGas", "Region",
    "DrivAgeGroup", "VehAgeGroup", "BonusMalusGroup",
    "DensityGroup", "VehPowerGroup"
  )

  freq_data_clean <- freq_data_clean %>%
    mutate(across(all_of(factor_vars), as.factor))

  freq_data_clean
}

#' Save processed data
save_processed_data <- function(data, output_path = "data/processed/freq_data_clean.rds") {
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(data, output_path)
  message("Processed data saved to ", output_path)
}

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

run_data_loading <- function(output_path = "data/processed/freq_data_clean.rds") {
  cat("=== DATASET OVERVIEW ===\n")
  raw <- load_raw_data()
  freq_data <- raw$freq
  sev_data <- raw$sev

  cat("Frequency dataset dimensions:", dim(freq_data), "\n")
  cat("Severity dataset dimensions:", dim(sev_data), "\n")

  total_claims <- sum(freq_data$ClaimNb)
  total_exposure <- sum(freq_data$Exposure)
  cat("\nTotal policies:", nrow(freq_data), "\n")
  cat("Total claims:", total_claims, "\n")
  cat("Overall claim frequency:", round(total_claims / total_exposure, 4), "\n")

  freq_data_clean <- preprocess_data(freq_data)
  cat("\nAfter filtering:", nrow(freq_data_clean), "records\n")

  save_processed_data(freq_data_clean, output_path)
  invisible(freq_data_clean)
}

