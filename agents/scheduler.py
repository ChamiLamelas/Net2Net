#!/usr/bin/env python3.8


import time


class Scheduler:
    def __init__(self, config):
        self.running_time = config["running_time"]
        self.gpu_changes = config["gpu_changes"]
        self.start_time = None
        self.change = 0

    def start(self):
        self.start_time = time.time()

    def timed_out(self):
        self.curr_time = time.time() - self.start_time
        return self.curr_time >= self.running_time

    def allocation(self):
        assert self.start_time is not None, "run start( ) first"
        self.curr_time = time.time() - self.start_time
        if self.curr_time >= self.gpu_changes[self.change]["time"]:
            self.change += 1
            return self.gpu_changes[self.change]["time"]["change"]
        return "same"


if __name__ == "__main__":
    pass
