# CRAN Comments

## Test environments

* local Ubuntu 22.04 install, R 4.5.2
* win-builder (devel and release)

## R CMD check results

There were no ERRORs or WARNINGs.

There is 1 NOTE:

* Found the following hidden files and directories:
  src/v/.../.cargo-checksum.json

These files are essential for the Rust `cargo` build system to verify the integrity of vendored dependencies. Removing them causes the build to fail. Please ignore this NOTE.
