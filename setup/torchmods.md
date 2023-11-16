To make torch modifications, example to InceptionNet, first find where the source code is with: 

```bash 
python3.8 -m pip show torchvision
```

Then, modify the source file, example: 

```bash
cp inception.py ~/.local/lib/python3.8/site-packages/torchvision/models/inception.py
```

Note, modifications to one file may affect others, for example modifications to `models/inception.py` can affect `models/quantization/inception.py`.
