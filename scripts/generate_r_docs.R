#!/usr/bin/env Rscript

lib_path <- file.path("package-r", "R_library")
if (dir.exists(lib_path)) {
  .libPaths(c(lib_path, .libPaths()))
}

if (!requireNamespace("roxygen2", quietly = TRUE)) {
  stop("roxygen2 is not installed. Please install it with install.packages('roxygen2')")
}

cat("Generating R documentation for 'package-r'...\n")
roxygen2::roxygenise("package-r")
cat("Documentation generation complete.\n")
