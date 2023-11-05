#!/usr/bin/env bash

echo THIS SCRIPT REQUIRES USER INPUT -- DONT LEAVE

# first part taken from here: https://www.itsupportwale.com/blog/how-to-upgrade-to-python-3-8-on-ubuntu-18-04-lts/

#sudo add-apt-repository ppa:deadsnakes/ppa
#sudo apt-get update
#apt-get update
#apt list | grep python3.8
#sudo apt-get install python3.8
#sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
#sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
#sudo update-alternatives --config python3

sudo apt install -y python3-pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install toml
python3 -m pip install matplotlib
python3 -m pip install tqdm
git config --global user.name ChamiLamelas
git config --global user.email chami.lamelas@gmail.com
