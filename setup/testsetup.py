import subprocess
import torch 
import sys 

output = subprocess.run(["nvidia-smi"], capture_output=True)
print("nvidia-smi")
print(output)

print("torch version")
print(torch.__version__)

print("python version")
print(sys.version)

print("cuda information")
print(torch.cuda.device_count())
