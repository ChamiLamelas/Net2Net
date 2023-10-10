import numpy as np 
import torch.nn as nn 
import torch 

def params_torch_to_tf_ndarr(layer, attr):
    param = getattr(layer, attr)
    assert isinstance(param, nn.Parameter)
    return np.transpose(param.data.numpy())


def params_tf_ndarr_to_torch(arr, layer, attr):
    setattr(layer, attr, nn.Parameter(torch.Tensor(np.transpose(arr))))

