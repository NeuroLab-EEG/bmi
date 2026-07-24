H5 := data/predictions.h5

.PHONY: all download evaluate metrics stats ma diagnostics carbon clean \
	check check-python check-r check-sql \
	fix fix-python fix-r fix-sql

all: stats

# Stage 0: download the 20 MOABB datasets to $DATA_PATH
download:
	python -m src.datasets

# Stage 1: run the full dataset x pipeline evaluation grid, writing to $DATA_PATH
# NOTE: writes predictions.h5/diagnostics/carbon logs under $DATA_PATH, not data/ --
# copy the finished files into data/ manually before running the stages below.
evaluate:
	python -m src.evaluation

# Stage 2: data/predictions.h5 -> data/predictions.csv + data/metrics.csv
data/metrics.csv: src/analysis/predictions-to-metrics.R $(H5)
	Rscript src/analysis/predictions-to-metrics.R $(H5) data

metrics: data/metrics.csv

# Stage 3: data/metrics.csv -> data/ma.csv
data/ma.csv: src/analysis/metrics-to-stats.sql data/metrics.csv
	duckdb -csv -c ".read src/analysis/metrics-to-stats.sql" > data/ma.csv

stats: data/ma.csv

# Stage 4: data/ma.csv -> metafor rma() summaries, forest/Baujat/influence figures
ma: data/ma.csv
	Rscript src/analysis/ma.R

# MCMC convergence summary: data/diagnostics.csv -> data/diagnostics_summary.csv
diagnostics:
	duckdb -csv -c ".read src/analysis/diagnostics.sql" > data/diagnostics_summary.csv

# Energy/carbon cost summary: data/carbon.csv -> data/carbon_summary.csv
carbon:
	duckdb -csv -c ".read src/analysis/carbon.sql" > data/carbon_summary.csv

clean:
	rm -f data/predictions.csv data/metrics.csv data/ma.csv

# --- Lint/format: check only, no files modified (matches CI) ---

check: check-python check-r check-sql

check-python:
	ruff format --check src
	ruff check src

check-r:
	Rscript -e 'styler::style_dir("src", dry = "fail")'
	Rscript -e 'lintr::lint_dir("src")'

check-sql:
	sqlfluff lint src

# --- Lint/format: auto-fix in place ---

fix: fix-python fix-r fix-sql

fix-python:
	ruff format src
	ruff check --fix src

# lintr has no auto-fix mode -- styler covers the formatting half only
fix-r:
	Rscript -e 'styler::style_dir("src")'

fix-sql:
	sqlfluff fix src
