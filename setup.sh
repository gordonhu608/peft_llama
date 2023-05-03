#!/bin/bash

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Make the installer executable
chmod +x Miniconda3-latest-Linux-x86_64.sh

# Run the installer (accept the license, use default installation location, and initialize Miniconda)
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Add Miniconda to PATH
echo "export PATH=$HOME/miniconda3/bin:$PATH" >> ~/.bashrc

# Source .bashrc to update PATH
source ~/.bashrc

bash 

conda init

bash

conda env create -f minienv.yaml

conda activate minigpt4

# Print a message to indicate the environment is ready
echo "Miniconda installed and 'minigpt4' environment is activated."




