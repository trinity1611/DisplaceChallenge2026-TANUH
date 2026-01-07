#! /bin/bash

CONDA_BASE=$(conda info --base) # or specify the path directly, e.g., ~/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"

#clone GitHub repository for DiariZen
git clone https://github.com/BUTSpeechFIT/DiariZen.git

cd DiariZen

# create virtual python environment
conda create --name diarizen python=3.9 --yes
conda activate diarizen

# install diarizen 
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia --yes
pip install -r requirements.txt && pip install -e .

# install pyannote-audio
cd pyannote-audio && pip install -e .[dev,testing]

# install dscore
git submodule init
git submodule update

cd ../../
mv inference_withConfigFile.py DiariZen/
