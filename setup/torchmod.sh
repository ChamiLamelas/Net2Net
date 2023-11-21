#!/bin/bash

cd ../instrumentation

# To find where python source code is for packages, do python3.8 -m pip show <packagename>
cp inception.py ~/.local/lib/python3.8/site-packages/torchvision/models/inception.py

# For some reason another InceptionNet file uses BasicConv2d so we have to fix that too
sed -i 's/inception_module.BasicConv2d/inception_module.InceptionNet2NetDeepenBlock/g' /users/slamel01/.local/lib/python3.8/site-packages/torchvision/models/quantization/inception.py
