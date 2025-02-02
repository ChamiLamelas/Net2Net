"""
NEEDSWORK document
"""

import torch

def move(device, *objs):
    return [obj.to(device) for obj in objs]


def get_device(device=0):
    if torch.cuda.is_available():
        if 0 <= device < torch.cuda.device_count():
            return torch.device(f'cuda:{device}')
        return None 
    return torch.device('cpu')