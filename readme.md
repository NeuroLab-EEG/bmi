# Bayesian Motor Imagery (BMI)

## Python version management

Follow [pyenv](https://github.com/pyenv/pyenv) installation instructions for recent release of Python.

## Python virtual environment

```bash
# Create new virtual environment
python -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install packages
pip install tensorflow[and-cuda]

# Deactivate
deactivate

# Save dependencies for reproducibility
pip freeze > requirements.txt

# Recreate an environment elsewhere
pip install -r requirements.txt

# List packages that are not dependencies of other packages
pip list --not-required
```

## Environment variables

Create `.env` file in root of git repository.

```bash
DATA_PATH=/path/to/data
```

## Cache database in background

```bash
nohup python -m classifiers.cache > output.log 2>&1 &
```

## Train classifiers

```bash
python -m classifiers.evaluate
```
