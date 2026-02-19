"""Timing helpers for examples."""
from dataclasses import dataclass, field
from typing import List
import time
import numpy as np


@dataclass
class TimedBlock:
    enabled: bool = True
    samples: List[float] = field(default_factory=list)
    last: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.last = time.perf_counter() - self._t0
        if self.enabled:
            self.samples.append(self.last)

    def mean_std_ms(self):
        if not self.samples:
            return None, None
        arr = np.asarray(self.samples, dtype=float)
        return arr.mean() * 1e3, arr.std() * 1e3
