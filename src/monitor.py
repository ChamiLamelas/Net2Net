import subprocess
import csv
from io import StringIO
from collections import defaultdict
import time
import math
from pathlib import Path
import os

USAGE_GPU = "gpu"
USAGE_MEM = "mem"


def get_gpu_usage():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pci.bus_id,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
    )
    csvstream = StringIO(result.stdout.decode("utf-8"))
    reader = csv.reader(csvstream)
    output = defaultdict(dict)
    for row in reader:
        output[row[0]][USAGE_GPU] = int(row[1])
        output[row[0]][USAGE_MEM] = int(row[2])
    return dict(output)


def get_gpu_names():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,pci.bus_id", "--format=csv,noheader,nounits"],
        capture_output=True,
    )
    csvstream = StringIO(result.stdout.decode("utf-8"))
    reader = csv.reader(csvstream)
    return {row[1].strip(): row[0] for row in reader}


class GPUMonitor:
    def __init__(self, frequency):
        self.frequency = frequency

    def start(self, secs):
        self.log = list()
        for i in range(math.ceil(secs / self.frequency)):
            if i > 0:
                time.sleep(self.frequency)
            self.log.append(get_gpu_usage())

    def get_history(self):
        history = defaultdict(lambda: defaultdict(list))
        for entry in self.log:
            for id, util in entry.items():
                for usage, val in util.items():
                    history[id][usage].append(val)
        return {k: dict(v) for k, v in history.items()}

    def save_history(self, folder):
        hist = self.get_history()
        for name, usage in hist.items():
            for usage_type, vals in usage.items():
                Path(
                    os.path.join(folder, f"{name.replace(':', '.')}.{usage_type}")
                ).write_text("\n".join(str(e) for e in vals) + "\n")

def main():
    monitor = GPUMonitor(0.5)
    monitor.start(10)
    print(monitor.get_history())

if __name__ == '__main__':
    main()