# CRAN Comments

## Test environments

* local Ubuntu 22.04 install, R 4.5.2
* win-builder (devel and release)

## R CMD check results

There were no ERRORs or WARNINGs.

There is 1 NOTE:

* The package source includes a vendored copy of Rust dependencies in `src/v`. This is necessary for offline compilation on CRAN/R-universe.
 Removing them causes the build to fail. Please ignore this NOTE.
