# SolarProphet

Irradiance forecasting using a sky-camera.  

## General Information

Work on this project is supported by the Industry-Academia partnership ["The Center for a Solar-Powered Future by 2050"](https://www.spf2050.org/)

## Getting Started

The included file `sample_config.py` should be renamed to `config.py` and should contain values for the neptune project name and api key. This file is included in the `.gitignore` manifest so that the API key is not committed to a repository by mistake.

Instructions assume a Windows OS with NVIDIA card. Mathods should be similar for other operating systems but may require some modifications. If running without a graphics card, the tensorflow installation will be different, and running the model will use the CPU.

Follow the instructions below once the prerequisites are met to create an environment with the required dependencies, or execute the `create_env.sh` script to automatically do the steps below for an NVIDIA GPU setup or `create_env_NoGPU.sh` for a CPU installation of TensorFlow.

### Prerequisites

* [git](https://git-scm.com/downloads)
* [anaconda distribution](https://www.anaconda.com/products/distribution)

### Create conda environment named `solarprophet` with `python 3.8`:

```bash
conda create -n solarprophet python=3.8
conda activate solarprophet
```

### Follow GPU Setup instructions [from TensorFlow](https://www.tensorflow.org/install/pip) 

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip
# Anything above 2.10 is not supported on the GPU on Windows Native
pip install "tensorflow<2.11" 
# Check tensorflow connection to GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Install remaining packages

```bash
pip install pandas seaborn Pillow scipy dask statsmodels tqdm neptune neptune-tensorflow-keras ipykernel joblib scikit-learn
# pytables is a bit finnicky, needs to be installed and upgraded
pip install --user --upgrade tables  
```

### Common Problems
