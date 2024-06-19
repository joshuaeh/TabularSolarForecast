# SolarProphet

Irradiance forecasting under data transmission constraints using features extracted from a sky-camera. Work completed by Joshua Hammond, Ricardo Lara, Michael Baldea, and Brian Korgel.

A CNN-LSTM model uses tabular features extracted from a sky camera, local meteorological measurements, and the clear sky model to forecast irradiance up to two-hours ahead at 10-minute intervals. Novel contributions include:  

- A data-parsimonious approach to irradiance forecasting in contrast to many contemporary models that require high dimensional inputs.  
- A noise model inspired by control theory which models the effect of random unmeasured disturbances or dropped features.  
- Multiple Irradiance representations. We show strong results when predicting irradiance as the deviation from the persistence prediction at the time of forecast. We believe this creates a normalizing effect allowing the model to learn what inputs are associated with _changes_ from the persistence assumption.  

## General Information

Work on this project is supported by the Industry-Academia partnership ["The Center for a Solar-Powered Future by 2050"](https://www.spf2050.org/)

## Getting Started

The included file `sample_config.py` should be renamed to `config.py` and should contain values for the neptune project name and api key. This file is included in the `.gitignore` manifest so that the API key is not committed to a repository by mistake.

Instructions assume a Windows OS with NVIDIA card. Mathods should be similar for other operating systems but may require some modifications. If running without a graphics card, the tensorflow installation will be different, and running the model will use the CPU.

Follow the instructions below once the prerequisites are met to create an environment with the required dependencies, or execute the `create_env.sh` script to automatically do the steps below for an NVIDIA GPU setup or `create_env_NoGPU.sh` for a CPU installation of TensorFlow.

### Prerequisites

* [git](https://git-scm.com/downloads)
* [anaconda distribution](https://www.anaconda.com/products/distribution)

There may be problems with GPU dependencies on your system. These are the processes I used for my PC and M1 Mac, but your
results may vary. For specifics on installation, especially tensorflow, see:

* [Install Tensorflow Documentation](https://www.tensorflow.org/install/)
* [Tensorflow-metal plugin for Apple Silicon and AMD GPUs](https://developer.apple.com/metal/tensorflow-plugin/)

### Windows Installation

Create conda environment named `solarprophet` with `python 3.8`:

```bash
conda create -n solarprophet python=3.8
conda activate solarprophet
```

Follow GPU Setup instructions [from TensorFlow](https://www.tensorflow.org/install/pip) 

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip
# Anything above 2.10 is not supported on the GPU on Windows Native
pip install "tensorflow<2.11" 
# Check tensorflow connection to GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Install remaining packages

```bash
pip install pandas seaborn Pillow scipy dask statsmodels tqdm neptune neptune-tensorflow-keras ipykernel joblib scikit-learn Jinja2
# pytables is a bit finnicky, needs to be installed and upgraded
pip install --user --upgrade tables  
```

### Linux/MacOS Installation

Create conda environment named `solarprophet` with `python 3.8`:

```bash
conda create -n solarprophet python=3.8
conda activate solarprophet
```

Install packages:

```bash
python -m pip install tensorflow
```

If using tensorflow-metal for Apple Silicon or AMD GPUs:

```bash
python -m pip install tensorflow-metal
```

Install remaining packages:

```bash
python -m pip  install pandas seaborn Pillow scipy dask statsmodels tqdm neptune neptune-tensorflow-keras ipykernel joblib scikit-learn Jinja2
conda install pytables
```

If tensorflow installation worked correctly, `python tf_test.py` should work and start a training bar

## Manifest
