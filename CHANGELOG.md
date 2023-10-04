# Change Log 

## Description

This has a list of the changes I have made to the original code to get it to run with. Overall,
this code does not seem to have been tested on CUDA GPU. It also has issues because PyTorch
is not backwards compatible.

## System Information

```
Operating System:
        Windows 10
Python version:
        3.9.6 (tags/v3.9.6:db3ff76, Jun 28 2021, 15:26:21) [MSC v.1929 64 bit (AMD64)]
PyTorch version:
        1.13.0+cu117
GPU:
        NVIDIA GeForce RTX 3060
CUDA Version:
        12.2
```

## Log 

**10/3/2023** 
- Implemented into `train_mnist.py` and `train_cifar10.py` the ability for a model
to adapt in the middle of training. This is basically letting a teacher train half
of time then expand and train from there.

**10/2/2023**
- Reran the experiment with [the bug fix](https://github.com/erogol/Net2Net/issues/3) recommended in GitHub issues.
- The above fix did improve performance of Net2Net. For MNIST, both student models start instantaneously closer to the teacher in performance. For CIFAR10, one of the student models still starts instantaneously lower than the teacher. 

**10/1/2023**
- Instrumented [train_mnist.py](examples/train_mnist.py) and [train_cifar10.py](examples/train_cifar10.py) with [my logging utilities](https://github.com/ChamiLamelas/UsefulPythonLibraries) so we can record learning and timings.
- [train_mnist.py](examples/train_mnist.py) runs in 50 minutes and [train_cifar10.py](examples/train_cifar10.py) runs in 35 minutes.
- Added [plot.py](examples/plot.py) to make plots. See `python plot.py -h` for usage. Note, [the cifar plots](examples/plots/cifar) is from the original repository. 
- Made similar compatibility fixes to [train_cifar10.py](examples/train_cifar10.py) as done with [train_mnist.py](examples/train_mnist.py).
- Increase number of epochs for both MNIST and CIFAR10 learning to allow models to actually learn to convergence. 
- Add observations and notes in [NOTES.md](NOTES.md).

**9/30/2023**
- Replace all instances of `.data[0]` with `.item()` on 1-element tensors (errors) in example code.
- Removed use of `pin_memory` for dataset transfer, caused a crash when running with CUDA GPU.
- Replaced use of `noise_var` with `noise=True` in [train_mnist.py](examples/train_mnist.py) (caused a `TypeError`).
- Transferred `nw1` and `nw2` to CPU before further NumPy computation.
- Fix up other PyTorch versioning warnings (e.g. `dim` in `log_softmax`).
- Modify [.gitignore](.gitignore) to not track things we don't want.
- Checked that with above changes the loss (accuracy) seems to decrease (increase) for MNIST learning.
