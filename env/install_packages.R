# Packages of the R labs of APM_50441_EP (Labs 2, 4 and 7), core and "To go further" parts.
#
# Usage (from R, or `Rscript install_packages.R lab2 lab4`):
#     source("install_packages.R")      # all the labs
#     install_lab("lab2")               # one lab
#     install_spark()                   # optional: Spark for Lab 7 (with winutils.exe on Windows)
#
# The packages are installed with {pak} when it is available (faster, binary packages, system
# requirements), otherwise with install.packages(). Only the missing packages are installed.
# On Colab (R runtime), the lab notebooks use the Posit Package Manager binaries instead (see their setup cell).

pkgs <- list(
  common = c("tidyverse", "patchwork", "knitr", "rmarkdown", "IRkernel"),
  lab2 = c(
    # core: simulated problem, plug-in, ERM, kNN, then the tidymodels workflow on TwoClass
    "tidymodels", "discrim", "naivebayes", "kknn", "glmnet", "kernlab", "brulee", "torch",
    # to go further: R AutoML, regularization paths, LOESS
    "finetune", "stacks", "mlr3", "mlr3learners", "mlr3pipelines", "mlr3tuning", "mlr3mbo", "e1071", "mirai", "ranger", "xgboost",
    "rpart", "future", "furrr"
  ),
  lab4 = c(
    # core: curse, PCA, nonlinear dimension reduction, clustering, autoencoder
    "cluster", "igraph", "dbscan", "mclust", "Rtsne", "uwot", "torch", "patchwork",
    # to go further: label propagation, blocked CV, ROC, MCA, ICA, EM, generative models
    "Matrix", "mvtnorm", "MASS", "ranger", "yardstick", "FactoMineR", "fastICA"
  ),
  lab7 = c(
    # core: I/O, split/apply/combine, profiling, parallelism, serving
    "data.table", "R.utils", "arrow", "duckdb", "duckplyr", "dbplyr", "DBI", "bench", "profvis", "lobstr", "Rcpp",
    "mirai", "carrier", "future", "furrr", "rsample", "butcher", "plumber2", "callr", "httr2", "bslib",
    "shiny", "bsicons", "testthat",
    # to go further: Spark, targets, in-database scoring, model versioning
    "sparklyr", "targets", "crew", "orbital", "tidypredict", "vetiver", "pins", "plumber"  # vetiver generates plumber (v1) APIs
  )
)

install_missing <- function(packages) {
  missing <- setdiff(unique(packages), rownames(installed.packages()))
  if (length(missing) == 0) {
    message("All the packages are already installed.")
    return(invisible(character(0)))
  }
  message("Installing: ", paste(missing, collapse = ", "))
  if (requireNamespace("pak", quietly = TRUE)) {
    pak::pkg_install(missing, ask = FALSE)
  } else {
    install.packages(missing)
  }
  invisible(missing)
}

install_lab <- function(lab = c("lab2", "lab4", "lab7")) {
  lab <- match.arg(lab, several.ok = TRUE)
  install_missing(c(pkgs$common, unlist(pkgs[lab])))
  # {torch} downloads its own C++ library (libtorch) the first time
  if (any(c("lab2", "lab4") %in% lab) && requireNamespace("torch", quietly = TRUE) && !torch::torch_is_installed()) {
    torch::install_torch()
  }
}

# Spark (only for the Spark block of Lab 7, about 400 MB, needs Java 17 or 21): not installed by install_lab().
# On Windows, Spark also needs Hadoop's winutils.exe, which is not shipped with Spark: it is downloaded into the
# folder tmp/hadoop/bin of the Spark installation, where sparklyr looks for it (no HADOOP_HOME to set).
install_spark <- function() {
  install_missing("sparklyr")
  if (nrow(sparklyr::spark_installed_versions()) == 0) sparklyr::spark_install()
  if (.Platform$OS.type == "windows") {
    winutils_dir <- file.path(sparklyr::spark_home_dir(), "tmp", "hadoop", "bin")
    if (!file.exists(file.path(winutils_dir, "winutils.exe"))) {
      dir.create(winutils_dir, recursive = TRUE, showWarnings = FALSE)
      download.file("https://github.com/steveloughran/winutils/raw/master/hadoop-3.0.0/bin/winutils.exe",
                    file.path(winutils_dir, "winutils.exe"), mode = "wb")
    }
  }
}

if (sys.nframe() == 0L) {  # run as a script
  args <- commandArgs(trailingOnly = TRUE)
  install_lab(if (length(args)) args else c("lab2", "lab4", "lab7"))
}
