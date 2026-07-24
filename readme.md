# Bayesian Motor Imagery (BMI)

## Python Guide

### Conda setup

```bash
# Create environment
conda create -n evaluation python=3.12

# Delete environment
conda env remove -n evaluation

# Clone environment
conda create --name new_evaluation --clone evaluation

# List environments
conda env list

# Activate environment
conda activate evaluation

# Deactivate environment
conda deactivate

# Save environment
conda env export > evaluation.yml

# Recreate environment
conda env create -f evaluation.yml
```

#### GPU stack

```bash
# Create activation script
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Create deactivation script
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'unset LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```

### Environment variables

Create `.env` file in root of git repository.

```bash
DATA_PATH=/path/to/data
RANDOM_STATE=1
```

### Background commands

```bash
# Start process
nohup python -m path.to.command > output.log 2>&1 &

# Find process
ps aux | grep "python -m path.to.command"

# Kill processes by username and full command
pkill -u username -f "substring"
```

### File format & linting

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

## R Guide

### Conda setup

```bash
# Create environment
conda create -n analysis -c conda-forge r-base

# Install system dependencies
conda install -c conda-forge \
    fontconfig \
    harfbuzz \
    fribidi \
    freetype \
    libpng \
    libtiff \
    libjpeg-turbo \
    zstd \
    libgcc-ng \
    compilers \
    zlib \
    pkg-config
```

#### Environment automation

```bash
# Create activation script
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/r_env_vars.sh << 'EOF'
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export CPPFLAGS="-I$CONDA_PREFIX/include"
export LDFLAGS="-L$CONDA_PREFIX/lib"
EOF

# Create deactivation script
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cat > $CONDA_PREFIX/etc/conda/deactivate.d/r_env_vars.sh << 'EOF'
unset CPPFLAGS
unset LDFLAGS
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s|$CONDA_PREFIX/lib:||")
export PKG_CONFIG_PATH=$(echo $PKG_CONFIG_PATH | sed "s|$CONDA_PREFIX/lib/pkgconfig:||")
EOF
```

### Renv project

```bash
# Enter R shell
R

# Install renv
install.packages("renv")

# Create renv
renv::init()

# Install packages
install.packages("tidyverse")

# Save renv
renv::snapshot()

# Run script
source("/path/to/software")

# Exit R shell
q()

## Recreate renv
renv::restore()
```

### File format & linting

```bash
# Within R shell
styler::style_dir("/path/to/software")
lintr::lint_dir("/path/to/software")
```

## SQL Guide

### File format & linting

```bash
# Lint files
sqlfluff format /path/to/software

# Fail if files not linted
sqlfluff lint /path/to/software
```
