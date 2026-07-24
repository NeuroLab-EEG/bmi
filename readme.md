# Bayesian Motor Imagery (BMI)

Code and analysis pipeline supporting the paper *"Bayesian Complete-Pooling in Cross-Subject Classification for Motor Imagery Electroencephalogram"*
[FILL IN: citation once accepted].

The paper asks whether Bayesian complete-pooling — averaging predictions over the
posterior rather than optimizing a single point estimate — meaningfully improves
calibration or discrimination for cross-subject motor imagery EEG classification,
and at what computational cost. This repository is the full pipeline behind that
answer: it downloads 20 public left/right hand motor imagery EEG datasets via
[MOABB](https://moabb.neurotechx.com), fits 12 pipelines (6 frequentist baselines
each paired with a Bayesian pipeline sharing identical feature engineering) under
cross-subject leave-one-subject-out or 10-fold cross-validation, then runs the full
random-effects meta-analysis across datasets.

| Category       | Frequentist | Bayesian |
| -------------- | ----------- | -------- |
| Raw signal     | CSP+LDA     | CSP+BLDA |
| Raw signal     | CSP+SVM     | CSP+GP   |
| Riemannian     | TS+LR       | TS+BLR   |
| Riemannian     | TS+SVM      | TS+GP    |
| Deep learning  | SCNN        | BSCNN    |
| Deep learning  | DCNN        | BDCNN    |

## Repository structure

```
src/
├── datasets/       # MOABB dataset download
├── pipelines/
│   ├── raw_signal/    # CSPLDA, CSPBLDA, CSPSVM, CSPGP
│   ├── riemannian/    # TSLR, TSBLR, TSSVM, TSGP
│   ├── deep_learning/ # SCNN, BSCNN, DCNN, BDCNN
│   └── classifiers/   # shared model-builder, neural-network, and subprocess base classes
├── evaluation/     # cross-subject evaluation loop
└── analysis/       # R/SQL: predictions -> metrics -> meta-analysis -> figures

data/               # analysis outputs: metrics, meta-analysis tables, figures, reports
patches/            # patch for the MOABB dependency below
```

## MOABB dependency

MOABB discards individual trial predictions and fitted models by default, keeping
only summary scikit-learn metrics. `patches/moabb-predictions.patch`
adds an opt-in `save_predictions` flag to `CrossSubjectEvaluation` that writes
per-fold predictions to `predictions.h5` alongside the usual metrics. It applies on
top of [davisethan/moabb](https://github.com/davisethan/moabb) at commit
`8391a6e1eda9de5c12961d8872f5eda9f0f59a1f`:

```bash
git clone https://github.com/davisethan/moabb.git
cd moabb && git checkout 8391a6e1eda9de5c12961d8872f5eda9f0f59a1f
git apply /path/to/bmi/patches/moabb-predictions.patch
pip install -e .
```

## Setup

### Python

```bash
conda env create -f environment.yml
conda activate evaluation
```

If training on GPU, the CUDA libraries conda installs need to be on the loader path – conda doesn't wire this up automatically:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'unset LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```

Create a `.env` file in the repository root:

```bash
DATA_PATH=/path/to/data
RANDOM_STATE=1
```

### R

The `renv` project (`renv.lock`, `.Rprofile`, `renv/`) covers `src/analysis`'s
dependencies:

```bash
Rscript -e 'renv::restore()'
```

## Running

`make stats` (the default `all` target) runs `download` through `stats` in sequence
given the prerequisite files already exist; each stage can also be run standalone.

```bash
make download      # fetch all 20 datasets
make evaluate      # run the full dataset x pipeline evaluation grid
make metrics       # data/predictions.h5 -> data/predictions.csv + data/metrics.csv
make stats         # data/metrics.csv -> data/ma.csv
make ma            # data/ma.csv -> rma() summaries, forest/Baujat/influence figures
make diagnostics   # data/diagnostics.csv -> data/diagnostics_summary.csv
make carbon        # data/carbon.csv -> data/carbon_summary.csv
```

## Data availability

Raw per-run outputs (`predictions.h5`, per-dataset/pipeline `scores.csv`, CodeCarbon
`emissions/` logs) are archived on Zenodo: https://doi.org/10.5281/zenodo.21538705.
The consolidated files
derived from them (`metrics.csv`, `ma.csv`, `carbon.csv`, `diagnostics.csv`, figures,
reports) are tracked directly in `data/` above.

## Development

```bash
make check   # check-python, check-r, check-sql -- no files modified
make fix     # fix-python, fix-r, fix-sql -- rewrites files in place
```
