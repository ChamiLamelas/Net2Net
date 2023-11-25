#!/usr/bin/env python3.8

from pathlib import Path
import multiprocessing
import subprocess
import numpy as np
import torch
import time
import os 


def move(device, *objs):
    return [obj.to(device) for obj in objs]


def get_device(device=0):
    if torch.cuda.is_available():
        if 0 <= device < torch.cuda.device_count():
            return torch.device(f"cuda:{device}")
        return None
    return torch.device("cpu")


class GPUMonitor:
    def __init__(self, frequency=1, logfile="gpu.log", windowsize=10, device=0):
        self.frequency = frequency
        self.logfile = logfile
        self.monitor_process = multiprocessing.Process(target=self.get_gpu_usage)
        self.window = list()
        self.windowsize = windowsize
        self.device = device
        Path(self.logfile).write_text("")

    def get_gpu_usage(self):
        while True:
            ti = time.time()
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            ).decode("utf-8")
            for line in result.splitlines():
                query = line.split(",")
                if int(query[0]) == self.device:
                    self.window.append(float(query[1]) / 100.0)
                    break
            if len(self.window) > self.windowsize:
                self.window.pop(0)
            with open(self.logfile, mode="a+", encoding="utf-8") as f:
                f.write(f"{np.mean(self.window):4f}\n")
            runtime = time.time() - ti
            if self.frequency > runtime:
                time.sleep(self.frequency - runtime)

    def query(self):
        contents = Path(self.logfile).read_text().splitlines()
        if len(contents) == 0:
            raise RuntimeError("No GPU information has been logged")
        return float(contents[-1])

    def start(self):
        self.monitor_process.start()

    def stop(self):
        self.monitor_process.terminate()
        os.remove(self.logfile)


if __name__ == "__main__":
    monitor = GPUMonitor()
    monitor.start()
    for i in range(10):
        time.sleep(2)
        print(monitor.query())
    monitor.stop()
