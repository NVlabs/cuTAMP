# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import torch


class TorchTimer:
    """Timer that synchronizes with GPU before recording time."""

    def __init__(self):
        # Dict of metric name to list of times
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._start_time: Dict[str, float] = {}

    def has_timer(self, metric: str) -> bool:
        return metric in self._start_time

    def start(self, metric: str) -> None:
        if metric in self._start_time:
            raise ValueError(f"Timer already started for {metric}")
        torch.cuda.synchronize()
        self._start_time[metric] = time.perf_counter()

    def elapsed(self, metric: str) -> float:
        if metric not in self._start_time:
            raise ValueError(f"Timer not started for {metric}")
        torch.cuda.synchronize()
        return time.perf_counter() - self._start_time[metric]

    def stop(self, metric: str) -> float:
        if metric not in self._start_time:
            raise ValueError(f"Timer not started for {metric}")
        torch.cuda.synchronize()
        duration = time.perf_counter() - self._start_time[metric]
        self._metrics[metric].append(duration)
        del self._start_time[metric]
        return duration

    @contextmanager
    def time(self, metric: str, log_callback=None):
        self.start(metric)
        yield
        duration = self.stop(metric)
        if log_callback is not None:
            log_callback(f"{metric} took {duration:.2f}s")

    def get_summary(self, metric: str) -> Dict[str, float]:
        """Get summary timing statistics for a given metric."""
        if metric not in self._metrics:
            raise ValueError(f"Metric {metric} not found")
        times = np.array(self._metrics[metric])
        summary = {
            "total": float(times.sum()),
            "mean": float(times.mean()),
            "median": float(np.median(times)),
            "std": float(times.std()),
            "count": len(times),
        }
        return summary

    def get_summaries(self) -> Dict[str, Dict[str, float]]:
        """Dump summaries for all the metrics."""
        return {metric: self.get_summary(metric) for metric in self._metrics}
