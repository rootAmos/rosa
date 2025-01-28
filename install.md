# Installation Guide

## Basic Requirements
Install the basic requirements using:
```
pip install -r requirements.txt
```

## pyOptSparse Installation via Conda

1. Create a new conda environment:
```
conda create -n your_env_name python=3.10
```

2. Activate the environment:
```
conda activate your_env_name
```


2. Configure conda channels:
```
conda config --add channels conda-forge
conda config --add channels anaconda
```

3. Install pyOptSparse and IPOPT:
```
conda install -c conda-forge pyoptsparse ipopt
```


### For Windows Users Only
If you're on Windows, you'll need to install an additional package:

```
conda install libpgmath
```

