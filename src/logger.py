"""
NEEDSWORK document
"""

from datetime import timedelta, datetime
from pathlib import Path
from math import ceil
import inspect
import logging
import time
import pytz
import sys
import os
import csv
import torch
from collections import Counter, defaultdict


def delete_files(*files):
    for file in files:
        if os.path.isfile(file):
            os.remove(file)


def clear_files(*files):
    for file in files:
        if os.path.isfile(file):
            Path(file).write_text("")


def add_extension(file, extension):
    return (file + extension) if not file.lower().endswith(extension) else file


def drop_extension(file):
    return file[: file.index(".")] if "." in file else file


def get_caller():
    return inspect.stack()[1].function


def nice_seconds_string(seconds):
    return str(timedelta(seconds=ceil(seconds)))


def curr_time_est(format):
    dt = datetime.now(pytz.timezone("US/Eastern"))
    return dt if format is None else dt.strftime(format)


class MyTimerException(Exception):
    pass


class MyTimer:
    def __init__(self, stream=sys.stdout):
        self.stream = stream
        self.start_time = None
        self.stop_time = None
        self.task = None

    def start(self, task=None):
        if self.start_time is not None:
            raise MyTimerException("start( ) called twice with no stop( ) in between")
        self.task = (task + " ") if task is not None else ""
        start_str = f"{self.task}start time: {curr_time_est('%m/%d/%Y %I:%M:%S %p')}"
        if self.stream is not None:
            print(start_str, file=self.stream)
        self.start_time = time.time()
        return start_str

    def stop(self):
        self.stop_time = time.time()
        if self.start_time is None:
            raise MyTimerException("stop( ) called without calling start( )")
        runtime_str = f"{self.task}runtime (h:mm:ss): {nice_seconds_string(time.time() - self.start_time)}"
        if self.stream is not None:
            print(runtime_str, file=self.stream)
        self.start_time = None
        return runtime_str


class TimedLoggerException(Exception):
    pass


class TimedLogger:
    def _check_log_ok(self):
        if self.timer.start_time is None:
            raise TimedLoggerException(
                f"{get_caller}( ) called without calling start( )"
            )

    def __init__(self, log_folder=os.getcwd(), persist=True):
        if not os.path.isdir(log_folder):
            Path(log_folder).mkdir(parents=True)
        self.timer = MyTimer(stream=None)
        self.log_folder = log_folder
        self.logger = logging.getLogger("log")
        self.logger.setLevel(logging.DEBUG)
        self.persist = persist

    def start(self, log_file=None, task=None):
        if log_file is None:
            log_file = curr_time_est("%Y%m%d_%H%M%S") if task is None else task
        self.log_file = add_extension(os.path.join(self.log_folder, log_file), ".log")
        if not self.persist:
            delete_files(self.log_file)
        self.logger.handlers.clear()
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s | [%(levelname)s] : %(message)s"
        )
        fh.setFormatter(file_formatter)
        console_formatter = logging.Formatter("[%(levelname)s] : %(message)s")
        ch.setFormatter(console_formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info(self.timer.start(task))

    def stop(self):
        self.logger.info(self.timer.stop())

    def info(self, message):
        self._check_log_ok()
        self.logger.info(message)

    def debug(self, message):
        self._check_log_ok()
        self.logger.debug(message)

    def warn(self, message):
        self._check_log_ok()
        self.logger.warning(message)

    def error(self, message):
        self._check_log_ok()
        self.logger.error(message)


class ML_Logger(TimedLogger):
    def __init__(self, log_folder=os.getcwd(), persist=True):
        super().__init__(log_folder, persist)
        self.best_save_metric = None
        self.start_time = None
        self.counts = defaultdict(Counter)

    def start(self, log_file, metrics_file, task=None):
        super().start(log_file, task)
        self.metrics_file = os.path.join(self.log_folder, metrics_file)
        self.start_time = time.time()

    def log_metrics(self, metric, granularity, model=None):
        save_key = list(metric.keys())[0]
        save_metric = list(metric.values())[0]
        save_time = time.time() - self.start_time
        with open(
            self.metrics_file + "." + granularity + "." + save_key,
            mode="a+",
            encoding="utf-8",
        ) as f:
            f.write(f"{save_time},{save_metric}\n")
        self.counts[granularity][save_key] += 1
        if granularity == "epoch" and model is not None:
            if self.best_save_metric is None or save_metric > self.best_save_metric:
                torch.save(
                    model.state_dict(),
                    os.path.join(self.log_folder, f"bestmodel.pt"),
                )
                self.best_save_metric = save_metric
            torch.save(
                model.state_dict(),
                os.path.join(self.log_folder, f"model_after_{self.counts['epoch'][save_key]}_epochs.pt"),
            )

    @staticmethod
    def load_metrics(log_folder, metrics_file, metric, granularity):
        metrics_file = os.path.join(log_folder, metrics_file)

        times = list()
        metrics = list()

        with open(
            metrics_file + "." + granularity + "." + metric,
            mode="r",
            encoding="utf-8",
        ) as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                times.append(float(row[0]))
                metrics.append(float(row[1]))

        return {"times": times, "metrics": metrics}
