#!/bin/bash

# Create the conda environment with Python 3.8
conda create --name crowd_counting python=3.8 -y

echo "Created conda environment 'crowd_counting' with Python 3.8"

# Ensure conda is properly initialized for this shell session
. "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the new environment
conda activate crowd_counting

# Install specified libraries and packages
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch -y
pip install opencv-python scipy Pillow pandas matplotlib h5py

# Notify the user about activation after the script is done
echo "To activate this environment in the future, use:"
echo "conda activate crowd_count
