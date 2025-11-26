# Bayesian Motor Imagery (BMI)

## Python virtual environment

```bash
# Create new virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install packages
pip install numpy pandas scikit-learn

# Deactivate
deactivate

# Save dependencies for reproducibility
pip freeze > requirements.txt

# Recreate an environment elsewhere
pip install -r requirements.txt
```

## Environment variables

Create `.env` file in root of git repository.

```bash
DATA_PATH=/path/to/data
```
