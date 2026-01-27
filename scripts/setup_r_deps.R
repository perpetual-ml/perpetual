#!/usr/bin/env Rscript

lib_path <- file.path("package-r", "R_library")
if (!dir.exists(lib_path)) {
  dir.create(lib_path, recursive = TRUE)
}
.libPaths(c(lib_path, .libPaths()))

cat("Installing 'roxygen2' to", lib_path, "...\n")
install.packages("roxygen2", lib = lib_path, repos = "https://cloud.r-project.org")
cat("Installation complete.\n")
