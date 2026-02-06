#!/usr/bin/env Rscript

lib_path <- file.path("package-r", "R_library")
if (!dir.exists(lib_path)) {
  dir.create(lib_path, recursive = TRUE)
}
.libPaths(c(lib_path, .libPaths()))

cat("Installing dependencies to", lib_path, "...\n")
deps <- c("roxygen2", "testthat", "jsonlite")
install.packages(deps, lib = lib_path, repos = "https://cloud.r-project.org")
cat("Installation complete.\n")
