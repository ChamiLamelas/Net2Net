#!/usr/bin/env bash

sudo apt install -y python3-pip
python3 -m pip install -y torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install -y toml

git config --global user.name ChamiLamelas
git config --global user.email chami.lamelas@gmail.com
