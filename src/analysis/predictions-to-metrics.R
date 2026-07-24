#!/usr/bin/env Rscript
#
# Convert a MOABB-style predictions.h5 into publishable CSVs.
#
# Walks the HDF5 tree (<dataset>/<pipeline>/<subject>/<session>,
# each group holding y_true and y_pred_proba) and writes two files:
#
#   1. predictions.csv -- one row per trial (the raw data).
#   2. metrics.csv     -- one row per (dataset, pipeline, subject, session):
#                         n_trials, n_classes, auroc, entropy (sharpness),
#                         and brier/reliability/resolution/uncertainty via
#                         reliabilitydiag's CORP decomposition. No ECE.
#
# Usage: Rscript src/predictions-to-metrics.R [path/to/predictions.h5] [outdir]

suppressPackageStartupMessages({
  library(hdf5r)
  library(reliabilitydiag)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
h5_path <- if (length(args) >= 1) args[[1]] else "data"
outdir <- if (length(args) >= 2) args[[2]] else "data"

if (!file.exists(h5_path)) stop("File not found: ", h5_path)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# --------------------------------------------------------------------------
# Walk the HDF5 tree; collect every group that contains "y_true".
# --------------------------------------------------------------------------
find_prediction_groups <- function(grp, prefix = "") {
  found <- character(0)
  items <- grp$ls(recursive = FALSE)$name
  for (nm in items) {
    path <- if (prefix == "") nm else paste0(prefix, "/", nm)
    obj <- grp[[nm]]
    if (inherits(obj, "H5Group")) {
      if (obj$exists("y_true")) {
        found <- c(found, path)
      } else {
        found <- c(found, find_prediction_groups(obj, path))
      }
    }
  }
  found
}

parse_key <- function(key) {
  parts <- strsplit(key, "/", fixed = TRUE)[[1]]
  fields <- c("dataset", "pipeline", "subject", "session")
  out <- as.list(setNames(rep(NA_character_, 4), fields))
  for (i in seq_along(parts)) if (i <= 4) out[[fields[i]]] <- parts[i]
  out
}

# --------------------------------------------------------------------------
# Extract y_true / y_pred_proba from a group.
#
# y_pred_proba is stored -- and read by hdf5r -- as (n_classes, n_trials),
# the transpose of the numpy/h5py (n_trials, n_classes) convention, so it
# is transposed here immediately to keep every downstream computation in
# the more familiar row-per-trial shape.
# --------------------------------------------------------------------------
extract_group <- function(grp) {
  y_true <- grp[["y_true"]]$read()
  proba <- NULL
  if (grp$exists("y_pred_proba")) {
    raw <- grp[["y_pred_proba"]]$read()
    proba <- if (is.null(dim(raw))) raw else t(raw) # -> (n_trials, n_classes)
  }
  y_pred <- if (grp$exists("y_pred")) grp[["y_pred"]]$read() else NULL
  list(y_true = y_true, y_pred_proba = proba, y_pred = y_pred)
}

# --------------------------------------------------------------------------
# Raw per-trial rows.
# --------------------------------------------------------------------------
rows_for_predictions <- function(meta, data) {
  n <- length(data$y_true)
  df <- data.frame(
    dataset = meta$dataset, pipeline = meta$pipeline,
    subject = meta$subject, session = meta$session,
    trial_index = 0:(n - 1),
    y_true = data$y_true
  )
  if (!is.null(data$y_pred_proba)) {
    proba <- data$y_pred_proba
    if (is.null(dim(proba))) {
      df$y_pred_proba <- proba
    } else {
      for (c in seq_len(ncol(proba))) {
        df[[paste0("y_pred_proba_", c - 1)]] <- proba[, c]
      }
    }
  }
  if (!is.null(data$y_pred)) df$y_pred <- data$y_pred
  df
}

# --------------------------------------------------------------------------
# Metrics: AUROC (rank-based Mann-Whitney form, no extra dependency),
# Shannon entropy (sharpness), and Brier/reliability/resolution/uncertainty
# via reliabilitydiag's CORP decomposition (binary only).
# --------------------------------------------------------------------------
auroc_binary <- function(y, p1) {
  n_pos <- sum(y == 1)
  n_neg <- sum(y == 0)
  if (n_pos == 0 || n_neg == 0) {
    return(NA_real_)
  }
  r <- rank(p1)
  (sum(r[y == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

shannon_entropy_sharpness <- function(proba_matrix) {
  # Mean per-trial entropy, normalised to [0, 1] with base = n_classes.
  k <- ncol(proba_matrix)
  p <- pmax(proba_matrix, .Machine$double.eps) # guard log(0)
  row_entropy <- -rowSums(p * log(p, base = k))
  mean(row_entropy)
}

compute_metrics <- function(meta, data) {
  y <- data$y_true
  n_classes <- length(unique(y))
  result <- c(
    meta[c("dataset", "pipeline", "subject", "session")],
    list(
      n_trials = length(y), n_classes = n_classes,
      auroc = NA_real_, entropy = NA_real_, brier = NA_real_,
      reliability = NA_real_, resolution = NA_real_, uncertainty = NA_real_
    )
  )

  proba <- data$y_pred_proba
  if (is.null(proba)) {
    return(as.data.frame(result, stringsAsFactors = FALSE))
  }

  proba_matrix <- if (is.null(dim(proba))) cbind(1 - proba, proba) else proba
  result$entropy <- shannon_entropy_sharpness(proba_matrix)

  if (n_classes == 2 && ncol(proba_matrix) == 2) {
    p1 <- proba_matrix[, 2]
    result$auroc <- auroc_binary(y, p1)
    decomp <- tryCatch(
      {
        s <- summary(reliabilitydiag(p1 = p1, y = y, region.level = NA), score = "brier")
        list(
          brier = s$mean_score, reliability = s$miscalibration,
          resolution = s$discrimination, uncertainty = s$uncertainty
        )
      },
      error = function(e) NULL
    )
    if (!is.null(decomp)) {
      result$brier <- decomp$brier
      result$reliability <- decomp$reliability
      result$resolution <- decomp$resolution
      result$uncertainty <- decomp$uncertainty
    }
  }
  as.data.frame(result, stringsAsFactors = FALSE)
}

# --------------------------------------------------------------------------
f <- H5File$new(h5_path, mode = "r")
group_paths <- find_prediction_groups(f)
if (length(group_paths) == 0) stop("No groups containing 'y_true' were found.")
cat(sprintf("Found %d prediction groups.\n", length(group_paths)))

pred_rows <- vector("list", length(group_paths))
metric_rows <- vector("list", length(group_paths))
for (i in seq_along(group_paths)) {
  key <- group_paths[i]
  meta <- parse_key(key)
  data <- extract_group(f[[key]])
  pred_rows[[i]] <- rows_for_predictions(meta, data)
  metric_rows[[i]] <- compute_metrics(meta, data)
}
f$close_all()

pred_df <- bind_rows(pred_rows)
metric_df <- bind_rows(metric_rows)

pred_csv <- file.path(outdir, "predictions.csv")
metric_csv <- file.path(outdir, "metrics.csv")
write.csv(pred_df, pred_csv, row.names = FALSE, quote = FALSE)
write.csv(metric_df, metric_csv, row.names = FALSE, quote = FALSE)

cat(sprintf("Wrote %d trial rows -> %s\n", nrow(pred_df), pred_csv))
cat(sprintf("Wrote %d metric rows -> %s\n", nrow(metric_df), metric_csv))
print(head(metric_df, 10))
