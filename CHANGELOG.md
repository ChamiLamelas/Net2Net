# Change log 

This has a list of the changes I have made to the original code to get it to run with. Overall,
this code does not seem to have been tested on CUDA GPU. It also has issues because PyTorch
is not backwards compatible.

**10/1/2023**
- Instrumented `train_mnist.py` with [my logging utilities](https://github.com/ChamiLamelas/UsefulPythonLibraries) so we can record learning and times.
- `train_mnist.py` runs in 20 minutes on 1 NVIDIA GeForce RTX 3060.
- Added `plot.py` to make plots. See `python plot.py -h` for usage. Note, `plots/cifar` is from the original repository. 
- Made similar compatibility fixes to `train_cifar10.py` as done with `train_mnist.py`.

**9/30/2023**
- Replace all instances of `.data[0]` with `.item()` on 1-element tensors (errors) in example code.
- Removed use of `pin_memory` for dataset transfer, caused a crash when running with CUDA GPU.
- Replaced use of `noise_var` with `noise=True` in `train_mnist.py` (caused a `TypeError`).
- Transferred `nw1` and `nw2` to CPU before further NumPy computation.
- Fix up other PyTorch versioning warnings (e.g. `dim` in `log_softmax`).
- Modify `.gitignore` to not track things we don't want.
- Checked that with above changes the loss (accuracy) seems to decrease (increase) for MNIST learning.