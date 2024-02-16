#!/bin/bash
# Script to create env for solarprophet on commputer without GPU
# Author: Joshua Hammond

# TODO check for conda install, and warn if not installed
#!/bin/bash
# Script to create conda env for solarprophet on commputer with NVIDIA GPU
# Author: Joshua Hammond

# TODO check for conda install, and warn if not installed
# create conda env
echo "Creating conda env"
conda create -q -n solarprophet python=3.8 -y
conda activate solarprophet

# install tensorflow
printf  "Installing tensorflow"
pip install --upgrade pip -q
# Anything above 2.10 is not supported on the GPU on Windows Native
pip install "tensorflow<2.11" -q
# check tensorflow installation with the following command:
printf  "\nChecking TensorFlow installation. Check the output below:"
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

# install other required packages:
printf  "Installing Other Packages"
pip install -q pandas seaborn Pillow scipy dask statsmodels tqdm neptune neptune-tensorflow-keras ipykernel joblib scikit-learn Jinja2 --user --no-cache-dir --no-warn-script-location
# pytables is a bit finnicky, needs to be installed and upgraded
pip install -q --user --upgrade tables 

printf  "\n\nComplete."
printf  "Run 'conda activate solarprophet' to activate the environment."
printf  "You may close this window if you like."
bash
