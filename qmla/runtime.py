"""Per-run timing and sampled resident-memory measurements."""
from __future__ import annotations

import platform
import threading
import time
from datetime import datetime, timezone

import psutil

from qmla.utils import require_torch


def process_started_at_utc() -> str:
    """Return the OS process launch time, including time spent importing modules."""
    return datetime.fromtimestamp(psutil.Process().create_time(), timezone.utc).isoformat()


class RuntimeMonitor:
    def __init__(self, device):
        self.device = device
        self.peak_rss = 0
        self.stop_event = threading.Event()

    def _sample(self):
        process = psutil.Process()
        while not self.stop_event.is_set():
            total = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
            self.peak_rss = max(self.peak_rss, total)
            self.stop_event.wait(0.05)

    def __enter__(self):
        torch = require_torch()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        torch = require_torch()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self.started
        self.finished_at_utc = datetime.now(timezone.utc).isoformat()
        self.stop_event.set()
        self.thread.join()

    def results(self, samples: int) -> dict:
        torch = require_torch()
        return {
            "process_started_at_utc": process_started_at_utc(),
            "measurement_started_at_utc": self.started_at_utc,
            "measurement_finished_at_utc": self.finished_at_utc,
            "elapsed_seconds": self.elapsed,
            "samples_per_second": samples / self.elapsed if self.elapsed else 0,
            "peak_process_rss_bytes": self.peak_rss,
            "memory_measurement": "RSS sampled every 50 ms, including loader workers; shared pages may be counted more than once",
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "cpu_threads": torch.get_num_threads(),
            "system_ram_bytes": psutil.virtual_memory().total,
            "torch_cuda_version": torch.version.cuda,
        }
