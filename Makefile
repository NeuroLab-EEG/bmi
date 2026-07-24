H5 := data/predictions.h5

.PHONY: all download evaluate metrics stats ma diagnostics carbon clean \
	check check-python check-r check-sql \
	fix fix-python fix-r fix-sql

all: stats

download:
	python -m src.datasets

evaluate:
	python -m src.evaluation

data/metrics.csv: src/analysis/predictions-to-metrics.R $(H5)
	Rscript src/analysis/predictions-to-metrics.R $(H5) data

metrics: data/metrics.csv

data/ma.csv: src/analysis/metrics-to-stats.sql data/metrics.csv
	duckdb -csv -c ".read src/analysis/metrics-to-stats.sql" > data/ma.csv

stats: data/ma.csv

ma: data/ma.csv
	Rscript src/analysis/ma.R

diagnostics:
	duckdb -csv -c ".read src/analysis/diagnostics.sql" > data/diagnostics_summary.csv

carbon:
	duckdb -csv -c ".read src/analysis/carbon.sql" > data/carbon_summary.csv

clean:
	rm -f data/predictions.csv data/metrics.csv data/ma.csv

check: check-python check-r check-sql

check-python:
	ruff format --check src
	ruff check src

check-r:
	Rscript -e 'styler::style_dir("src", dry = "fail")'
	Rscript -e 'lintr::lint_dir("src")'

check-sql:
	sqlfluff lint src

fix: fix-python fix-r fix-sql

fix-python:
	ruff format src
	ruff check --fix src

# lintr has no auto-fix mode
fix-r:
	Rscript -e 'styler::style_dir("src")'

fix-sql:
	sqlfluff fix src
