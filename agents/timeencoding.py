import numpy as np 

def encode_time(time_seconds, period=400, num_dimensions=8):
    # period is the length of the cycle (e.g., 24 hours in seconds)
    time_encoded = [
        np.sin(2 * np.pi * time_seconds / period * i)
        for i in range(1, num_dimensions + 1)
    ]
    return np.array(time_encoded)