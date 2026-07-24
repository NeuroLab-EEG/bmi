library(metafor)

data <- read.csv("data/ma.csv")

metrics <- c("brier", "reliability", "resolution", "auroc", "entropy")

metric_labels <- c(
  brier = "Brier Score", reliability = "Reliability", resolution = "Resolution",
  auroc = "AUROC", entropy = "Shannon Entropy"
)

fig_dir <- "data/figures"
report_dir <- "data/reports"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

# Fits one metric's model and writes its summary/leave-one-out/influence text
# to data/reports/<metric>_summary.txt. sink(..., split = TRUE) mirrors
# everything to the console too, so nothing is lost from interactive use.
# The report connection is opened/closed inside this function (not at the
# top level) so on.exit gives it its own per-call scope -- guaranteeing the
# file gets closed even if rma()/influence() errors partway through, which
# a bare on.exit() inside a top-level for loop would not.
run_metric <- function(metric) {
  yi <- data[[paste0("yi_", metric)]]
  vi <- data[[paste0("vi_", metric)]]

  m <- rma(yi = yi, vi = vi, slab = data$dataset, method = "REML", test = "knha")
  inf <- influence(m)

  report_path <- file.path(report_dir, paste0(metric, "_summary.txt"))
  con <- file(report_path, open = "wt")
  sink(con, split = TRUE)
  on.exit(
    {
      sink()
      close(con)
    },
    add = TRUE
  )

  cat("\n============================================================\n")
  cat("Metric:", metric, "\n")
  cat("============================================================\n")
  print(summary(m))

  cat("\n--- Prediction interval ---\n")
  print(predict(m))

  cat("\n--- Leave-one-out ---\n")
  print(leave1out(m))

  cat("\n--- Influence diagnostics (rstudent, dffits, cook.d, hat, weight, ...) ---\n")
  print(inf)

  list(model = m, influence = inf)
}

models <- list()
for (metric in metrics) {
  result <- run_metric(metric)
  models[[metric]] <- result$model

  label <- metric_labels[[metric]]
  # Sized in real inches at print resolution (300+ DPI), not raw pixels --
  # R scales text relative to the device's inch dimensions, so drawing at
  # ~8in (close to how large these will actually appear on the page) keeps
  # text legible even if the figure is placed at less than \textwidth,
  # while the high DPI keeps it crisp regardless of display size.
  png(file.path(fig_dir, paste0(metric, "_forest.png")),
    width = 8, height = 8.4, units = "in", res = 350
  )
  forest(
    result$model,
    addpred = TRUE,
    digits = 4,
    xlab = paste0(label, " (Bayesian − Frequentist)"),
    mlab = paste0("Pooled Effect (", label, ")"),
    header = c("Dataset", "Estimate [95% CI]")
  )
  dev.off()

  png(file.path(fig_dir, paste0(metric, "_influence.png")),
    width = 8, height = 8, units = "in", res = 350
  )
  plot(result$influence)
  dev.off()

  # Plot all studies as plain points, then text-label only the "extreme"
  # ones (top quartile on either axis) -- with slab names on every point
  # the tightly-clustered, low-heterogeneity/low-influence studies near
  # the origin overlap into unreadable text. This is metafor's own
  # documented pattern for this exact problem (see ?baujat examples).
  png(file.path(fig_dir, paste0(metric, "_baujat.png")),
    width = 8, height = 7.1, units = "in", res = 350
  )
  sav <- baujat(result$model, symbol = 19, cex = 0.8)
  extreme <- sav[sav$x >= quantile(sav$x, 0.75) | sav$y >= quantile(sav$y, 0.75), ]
  text(extreme$x, extreme$y, extreme$slab, pos = 3, cex = 0.7)
  dev.off()
}
