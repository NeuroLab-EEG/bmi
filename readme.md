# Bayesian Motor Imagery (BMI)

## Conda setup

```bash
# Create environment
conda create -n bmi python=3.12

# Delete environment
conda env remove -n bmi

# Activate environment
conda activate bmi

# Deactivate environment
conda deactivate

# Save environment
conda env export > environment.yml

# Recreate environment
conda env create -f environment.yml
```

### GPU stack

```bash
# Create activation script
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Create deactivation script
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'unset LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```

## Environment variables

Create `.env` file in root of git repository.

```bash
DATA_PATH=/path/to/data
RANDOM_STATE=1
```

## Background commands

```bash
# Start process
nohup python -m path.to.command > output.log 2>&1 &

# Find process
ps aux | grep "python -m path.to.command"

# Kill processes by username and full command
pkill -u username -f "substring"
```

## File format & linting

```bash
# Format files
ruff format /path/to/software

# Fail if files not formatted
ruff format --check /path/to/software

# Lint files
ruff check --fix /path/to/software

# Fail if files not linted
ruff check /path/to/software
```
