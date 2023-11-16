To make torch modifications, example to InceptionNet, first find where the source code is with: 

```bash 
python3.8 -m pip show torchvision
```

Then, modify the source file, example: 

```bash
rm ~/.local/lib/python3.8/site-packages/torchvision/models/inception.py
nano ~/.local/lib/python3.8/site-packages/torchvision/models/inception.py
```

Then copy paste in the contents of `src/inception.py`.

Note, modifications to one file may affect others, for example modifications to `models/inception.py` can affect `models/quantization/inception.py`.
