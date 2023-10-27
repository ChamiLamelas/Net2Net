#!/usr/bin/env bash

sudo apt install python3-pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install toml

git config --global user.name ChamiLamelas
git config --global user.email chami.lamelas@gmail.com
