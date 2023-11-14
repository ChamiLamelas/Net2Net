# DEEPENING 

- We are unable to deal with dropout because multiple forward calls on a model using dropout during training will randomly 0-out different model parameters, see pytorch note: "Each channel will be zeroed out independently on every forward call." from here: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html. For testing we can disable the dropout
- There seems to be a bug with either NumPy or PyTorch on Windows because the same code works to deepen with < 10^(-5) error on Ubuntu and it has ~10^(-2) error on windows. 
- As in the Net2Net paper, we only support expansions on models that use ReLU activation. This is because ReLU has the special property that: ReLU(ReLU(x)) = ReLU(x). This is necessary to make the justification that Net2Net adaptations 
maintain instantaneous accuracy.

# WIDENING 

Notes: 

- This does not support linear to conv or conv to linear widening, one way 
to potentially solve this issue is to convert a linear layer to an
identical 1x1 convolutional layer (https://datascience.stackexchange.com/a/12833).

- Widening linear-linear, convolutional-convolutional and deepening linear layers
are taken from here: https://github.com/paengs/Net2Net

- We are unable to handle nonsequential layer execution order, e.g. here's 
a forward function of a InceptionNet block. Note branch3x3 isn't passed
into following layers but is concatenated at the end. Hence, we can't 
arbitrarily widen all layers in a network, we need to have some knowledge
of which layers we can widen -- not sure how to determine this automatically.

```python
def _forward(self, x: Tensor) -> List[Tensor]:
    branch3x3 = self.branch3x3(x)

    branch3x3dbl = self.branch3x3dbl_1(x)
    branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
    branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

    branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

    outputs = [branch3x3, branch3x3dbl, branch_pool]
    return outputs
```

# SETUP 

* Only supported on Linux (we use chmod, shebangs, etc.)

# IMAGENET 

* Doesn't support the test set. See comments in ImageNet setup, data.py
