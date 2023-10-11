import numpy as np
import torch.nn as nn
import torch

# look at comments of: https://stackoverflow.com/a/36223553 for TF weights shape 

def params_torch_to_tf_ndarr(torch_layer, attr):
    torch_ndarr = getattr(torch_layer, attr).data.numpy()
    if attr == "bias":
        return torch_ndarr
    elif attr == "weight":
        if isinstance(torch_layer, nn.Linear):
            return np.transpose(torch_ndarr)
        elif isinstance(torch_layer, nn.Conv2d):
            out_channels, in_channels, kernel_height, kernel_width = torch_ndarr.shape
            return np.reshape(
                torch_ndarr, (kernel_height, kernel_width, in_channels, out_channels)
            )


def params_tf_ndarr_to_torch(tf_ndarr, torch_layer, attr):
    if attr == "bias":
        setattr(torch_layer, attr, nn.Parameter(torch.Tensor(tf_ndarr)))
    elif attr == "weight":
        if isinstance(torch_layer, nn.Linear):
            new_in_features, new_out_features = tf_ndarr.shape
            setattr(
                torch_layer, attr, nn.Parameter(torch.Tensor(np.transpose(tf_ndarr)))
            )
            torch_layer.in_features = new_in_features
            torch_layer.out_features = new_out_features
        elif isinstance(torch_layer, nn.Conv2d):
            (
                kernel_height,
                kernel_width,
                new_in_channels,
                new_out_channels,
            ) = tf_ndarr.shape
            setattr(
                torch_layer,
                attr,
                nn.Parameter(
                    torch.Tensor(
                        np.reshape(
                            tf_ndarr,
                            new_out_channels,
                            new_in_channels,
                            kernel_height,
                            kernel_width,
                        )
                    )
                ),
            )
            torch_layer.in_channels = new_in_channels
            torch_layer.out_channels = new_out_channels
