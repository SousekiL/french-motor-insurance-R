# =============================================================================
# 02 - Exploratory Data Analysis
# French Motor Insurance - EDA and visualization
# Reference: DAV GitHub Repository, insurancerating package
# =============================================================================

library(ggplot2)
library(gridExtra)
library(corrplot)
library(RColorBrewer)
library(scales)
library(dplyr)

theme_set(theme_minimal(base_size = 12))

# -----------------------------------------------------------------------------
# 2.1 Claim Frequency Distribution
# -----------------------------------------------------------------------------

plot_claim_exposure_distribution <- function(data, output_dir = "output/figures") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  p1 <- ggplot(data, aes(x = ClaimNb)) +
    geom_bar(fill = "steelblue", alpha = 0.7) +
    scale_y_log10(labels = comma) +
    labs(
      title = "Claim Count Distribution",
      subtitle = "Note: Log scale on y-axis (typical Poisson distribution)",
      x = "Number of Claims",
      y = "Number of Policies (log scale)"
    ) +
    theme(plot.title = element_text(face = "bold"))

  p2 <- ggplot(data, aes(x = Exposure)) +
    geom_histogram(bins = 50, fill = "coral", alpha = 0.7, color = "black") +
    labs(
      title = "Exposure Distribution",
      subtitle = "Most policies have full year exposure",
      x = "Exposure (years)",
      y = "Number of Policies"
    )

  g <- gridExtra::arrangeGrob(p1, p2, ncol = 2)
  ggsave(file.path(output_dir, "01_claim_exposure_distribution.png"),
    plot = g, width = 12, height = 5, dpi = 300
  )
}

# -----------------------------------------------------------------------------
# 2.2 Risk Factor Analysis - Driver Age
# -----------------------------------------------------------------------------

plot_frequency_by_age <- function(data, output_dir = "output/figures") {
  age_analysis <- data %>%
    group_by(DrivAgeGroup) %>%
    summarise(
      Policies = n(),
      TotalClaims = sum(ClaimNb),
      TotalExposure = sum(Exposure),
      Frequency = TotalClaims / TotalExposure,
      .groups = "drop"
    )

  p <- ggplot(age_analysis, aes(x = DrivAgeGroup, y = Frequency)) +
    geom_col(fill = "#2E86AB", alpha = 0.8) +
    geom_text(aes(label = sprintf("%.4f", Frequency)),
      vjust = -0.5, size = 3.5
    ) +
    labs(
      title = "Claim Frequency by Driver Age Group",
      subtitle = "Young drivers (18-24) have significantly higher claim rates",
      x = "Driver Age Group",
      y = "Claim Frequency (claims per exposure year)"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  ggsave(file.path(output_dir, "02_frequency_by_age.png"), width = 10, height = 6, dpi = 300)
  list(plot = p, analysis = age_analysis)
}

# -----------------------------------------------------------------------------
# 2.3 Risk Factor Analysis - Vehicle Power
# -----------------------------------------------------------------------------

plot_frequency_by_power <- function(data, output_dir = "output/figures") {
  power_analysis <- data %>%
    group_by(VehPower) %>%
    summarise(
      Policies = n(),
      TotalClaims = sum(ClaimNb),
      TotalExposure = sum(Exposure),
      Frequency = TotalClaims / TotalExposure,
      .groups = "drop"
    ) %>%
    filter(Policies >= 100)

  p <- ggplot(power_analysis, aes(x = factor(VehPower), y = Frequency)) +
    geom_col(fill = "#A23B72", alpha = 0.8) +
    geom_smooth(aes(x = as.numeric(factor(VehPower))),
      method = "loess", se = TRUE, color = "red", linetype = "dashed"
    ) +
    labs(
      title = "Claim Frequency by Vehicle Power",
      subtitle = "Higher power vehicles show increased claim frequency",
      x = "Vehicle Power (Tax HP)",
      y = "Claim Frequency"
    )

  ggsave(file.path(output_dir, "03_frequency_by_power.png"), width = 10, height = 6, dpi = 300)
  list(plot = p, analysis = power_analysis)
}

# -----------------------------------------------------------------------------
# 2.4 Risk Factor Analysis - Bonus-Malus
# -----------------------------------------------------------------------------

plot_frequency_by_bonus_malus <- function(data, output_dir = "output/figures") {
  bonus_analysis <- data %>%
    group_by(BonusMalusGroup) %>%
    summarise(
      Policies = n(),
      TotalClaims = sum(ClaimNb),
      TotalExposure = sum(Exposure),
      Frequency = TotalClaims / TotalExposure,
      .groups = "drop"
    )

  p <- ggplot(bonus_analysis, aes(x = BonusMalusGroup, y = Frequency)) +
    geom_col(fill = "#06A77D", alpha = 0.8) +
    geom_text(aes(label = sprintf("%.4f", Frequency)),
      vjust = -0.5, size = 3
    ) +
    labs(
      title = "Claim Frequency by Bonus-Malus Level",
      subtitle = "Merit-rating system effectively differentiates risk",
      x = "Bonus-Malus Category",
      y = "Claim Frequency"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  ggsave(file.path(output_dir, "04_frequency_by_bonus_malus.png"), width = 10, height = 6, dpi = 300)
  list(plot = p, analysis = bonus_analysis)
}

# -----------------------------------------------------------------------------
# 2.5 Geographic Analysis
# -----------------------------------------------------------------------------

plot_frequency_by_area <- function(data, output_dir = "output/figures") {
  area_analysis <- data %>%
    group_by(Area) %>%
    summarise(
      Policies = n(),
      TotalClaims = sum(ClaimNb),
      TotalExposure = sum(Exposure),
      Frequency = TotalClaims / TotalExposure,
      AvgDensity = mean(Density),
      .groups = "drop"
    )

  p <- ggplot(area_analysis, aes(x = reorder(Area, -Frequency), y = Frequency)) +
    geom_col(aes(fill = AvgDensity), alpha = 0.8) +
    scale_fill_gradient(
      low = "#FFF7BC", high = "#D95F0E",
      name = "Avg Density",
      labels = comma
    ) +
    labs(
      title = "Claim Frequency by Geographic Area",
      subtitle = "Area codes represent different regions in France",
      x = "Area Code",
      y = "Claim Frequency"
    )

  ggsave(file.path(output_dir, "05_frequency_by_area.png"), width = 10, height = 6, dpi = 300)
  list(plot = p, analysis = area_analysis)
}

# -----------------------------------------------------------------------------
# 2.6 Correlation Matrix
# -----------------------------------------------------------------------------

plot_correlation_matrix <- function(data, output_dir = "output/figures") {
  num_vars <- c("DrivAge", "VehAge", "VehPower", "BonusMalus", "LogDensity", "ClaimNb", "Frequency")
  cor_data <- data %>%
    select(all_of(num_vars)) %>%
    na.omit()
  cor_matrix <- cor(cor_data)

  png(file.path(output_dir, "06_correlation_matrix.png"), width = 800, height = 800)
  corrplot(cor_matrix,
    method = "color", type = "upper", addCoef.col = "black",
    tl.col = "black", tl.srt = 45, number.cex = 0.8,
    col = colorRampPalette(c("#6D9EC1", "white", "#E46726"))(200),
    title = "Correlation Matrix of Key Variables",
    mar = c(0, 0, 2, 0)
  )
  dev.off()

  cor_matrix
}

# -----------------------------------------------------------------------------
# 2.7 Risk Segment Analysis
# -----------------------------------------------------------------------------

plot_risk_segment_analysis <- function(data, output_dir = "output/figures") {
  risk_segment_analysis <- data %>%
    mutate(
      RiskSegment = case_when(
        DrivAge < 25 & VehPower > 9 ~ "Young + Powerful",
        DrivAge < 25 & VehPower <= 9 ~ "Young + Normal",
        DrivAge >= 25 & VehPower > 9 ~ "Mature + Powerful",
        TRUE ~ "Mature + Normal"
      )
    ) %>%
    group_by(RiskSegment) %>%
    summarise(
      Policies = n(),
      TotalClaims = sum(ClaimNb),
      TotalExposure = sum(Exposure),
      Frequency = TotalClaims / TotalExposure,
      .groups = "drop"
    ) %>%
    arrange(desc(Frequency))

  p <- ggplot(risk_segment_analysis, aes(x = reorder(RiskSegment, -Frequency), y = Frequency)) +
    geom_col(aes(fill = Frequency), alpha = 0.8, show.legend = FALSE) +
    scale_fill_gradient(low = "#90EE90", high = "#FF6347") +
    geom_text(aes(label = sprintf("%.4f", Frequency)), vjust = -0.5, fontface = "bold") +
    labs(
      title = "Claim Frequency by Risk Segment",
      subtitle = "Interaction effect: Young drivers with powerful cars are highest risk",
      x = "Risk Segment",
      y = "Claim Frequency"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  ggsave(file.path(output_dir, "07_risk_segment_analysis.png"), width = 10, height = 6, dpi = 300)
  list(plot = p, analysis = risk_segment_analysis)
}

# -----------------------------------------------------------------------------
# Main EDA runner
# -----------------------------------------------------------------------------

run_eda <- function(data = NULL, data_path = "data/processed/freq_data_clean.rds",
                    output_dir = "output/figures") {
  if (is.null(data)) {
    data <- readRDS(data_path)
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  plot_claim_exposure_distribution(data, output_dir)
  age_result <- plot_frequency_by_age(data, output_dir)
  power_result <- plot_frequency_by_power(data, output_dir)
  bonus_result <- plot_frequency_by_bonus_malus(data, output_dir)
  area_result <- plot_frequency_by_area(data, output_dir)
  cor_matrix <- plot_correlation_matrix(data, output_dir)
  segment_result <- plot_risk_segment_analysis(data, output_dir)

  summary_stats <- list(
    overall = data.frame(
      Metric = c("Total Policies", "Total Claims", "Total Exposure", "Overall Frequency"),
      Value = c(
        nrow(data),
        sum(data$ClaimNb),
        round(sum(data$Exposure), 2),
        round(sum(data$ClaimNb) / sum(data$Exposure), 4)
      )
    ),
    by_age = age_result$analysis,
    by_power = power_result$analysis,
    by_bonus = bonus_result$analysis,
    by_area = area_result$analysis,
    by_segment = segment_result$analysis
  )

  dir.create("output/tables", recursive = TRUE, showWarnings = FALSE)
  saveRDS(summary_stats, "output/tables/eda_summary_statistics.rds")

  cat("EDA complete! All figures saved to", output_dir, "\n")
  invisible(summary_stats)
}
