# =============================================================================
# Helper Functions
# French Motor Insurance - Utility functions
# =============================================================================

#' Ensure directory exists, create if not
#'
#' @param path Character. Directory path
ensure_dir <- function(path) {
  dir.create(path, showWarnings = FALSE, recursive = TRUE)
  invisible(path)
}

#' Set project root and working directory
#'
#' @param root Character. Project root path (default: parent of R script location)
set_project_root <- function(root = NULL) {
  if (is.null(root)) {
    root <- normalizePath(file.path(getwd(), ".."))
  }
  setwd(root)
  invisible(root)
}

#' Safe division (avoid division by zero)
#'
#' @param x Numeric
#' @param y Numeric
#' @return x / y or 0 if y == 0
safe_divide <- function(x, y) {
  ifelse(y == 0, 0, x / y)
}
