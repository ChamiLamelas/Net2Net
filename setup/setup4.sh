#!/usr/bin/env bash

echo THIS SCRIPT REQUIRES USER INPUT -- DONT LEAVE

# 1st part taken from here: https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04/
# the default python version is 3.6

sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.8

# this I figured out was necessary because otherwise torch would appear to install then show up
# nowhere in the system whether you run pip show/list on any of the installed pip versions
python3.8 -m pip install -U pip

# this first step is taken from here: https://pytorch.org/get-started/locally/
python3.8 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3.8 -m pip install toml matplotlib tqdm pytz
git config --global user.name ChamiLamelas
git config --global user.email chami.lamelas@gmail.com
chmod +x testsetup.py ../src/run.py ../tests/test*.py ../plotting/*.py
