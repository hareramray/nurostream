"""Async block reader.

Windows has no pread, so each worker thread keeps its own file handle and does
seek+read. Python releases the GIL for the duration of a read, so N workers
give genuine NVMe queue depth — which is the entire game: a single blocking
reader gets ~600 MB/s out of this drive, 16 concurrent readers get 2.35 GB/s.

Reads below MIN_READ are widened. This SSD is DRAM-less and small random reads
fall off a cliff; a 4 KB read costs nearly as much as a 256 KB one.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MIN_READ = 256 * 1024
# Measured on the target drive with 5.85 MB random reads (one MoE expert
# slab): 1 worker 0.62 GB/s, 8 workers 2.00, 24 workers 1.57, 48 workers 1.37.
# Queue depth helps up to a point and then hurts - a DRAM-less controller
# thrashes its limited mapping cache when too many streams compete. More
# threads is not more throughput; measure before raising this.
DEFAULT_WORKERS = 8


@dataclass
class IOStats:
    requests: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0
    stall_seconds: float = 0.0  # time the compute thread blocked on a future
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    first_read: float = 0.0
    last_read: float = 0.0

    @property
    def wall_seconds(self) -> float:
        return max(0.0, self.last_read - self.first_read)

    @property
    def throughput_gbps(self) -> float:
        """Wall-clock throughput.

        Dividing by summed per-thread read time would report ~0.27 GB/s for
        24 concurrent readers that are actually saturating the drive, because
        their waits overlap. Wall clock is the number that matters.
        """
        if self.wall_seconds <= 0:
            return 0.0
        return self.bytes_read / 1e9 / self.wall_seconds

    @property
    def prefetch_rate(self) -> float:
        total = self.prefetch_hits + self.prefetch_misses
        return 0.0 if total == 0 else self.prefetch_hits / total

    def summary(self) -> str:
        return (
            f"{self.bytes_read / 1e9:.2f}GB in {self.requests} reads | "
            f"{self.throughput_gbps:.2f} GB/s wall | "
            f"stall {self.stall_seconds:.2f}s | "
            f"prefetch {self.prefetch_rate:.1%}"
        )


class AsyncReader:
    """Thread-pool positional reader with per-thread file handles."""

    def __init__(self, path: str | Path, n_workers: int = DEFAULT_WORKERS) -> None:
        self.path = Path(path)
        self.n_workers = n_workers
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="ns-io"
        )
        self._handles: list = []
        self._handles_lock = threading.Lock()
        self.stats = IOStats()
        self._stats_lock = threading.Lock()

    def _fh(self):
        fh = getattr(self._local, "fh", None)
        if fh is None:
            fh = open(self.path, "rb", buffering=0)
            self._local.fh = fh
            with self._handles_lock:
                self._handles.append(fh)
        return fh

    def _read_blocking(self, offset: int, nbytes: int) -> torch.Tensor:
        t0 = time.perf_counter()
        fh = self._fh()
        fh.seek(offset)
        buf = fh.read(nbytes)
        dt = time.perf_counter() - t0
        if len(buf) != nbytes:
            raise EOFError(
                f"short read at {offset}: wanted {nbytes}, got {len(buf)}"
            )
        with self._stats_lock:
            self.stats.requests += 1
            self.stats.bytes_read += nbytes
            self.stats.read_seconds += dt
            now = time.perf_counter()
            if self.stats.first_read == 0.0:
                self.stats.first_read = t0
            self.stats.last_read = now
        return torch.from_numpy(np.frombuffer(buf, dtype=np.uint8).copy())

    def submit(self, offset: int, nbytes: int) -> Future:
        """Queue a read. Returns immediately."""
        return self._pool.submit(self._read_blocking, offset, nbytes)

    def submit_many(self, spans: list[tuple[int, int]]) -> list[Future]:
        return [self.submit(off, n) for off, n in spans]

    def read(self, offset: int, nbytes: int) -> torch.Tensor:
        """Blocking read on the calling thread."""
        return self._read_blocking(offset, nbytes)

    def wait(self, fut: Future) -> torch.Tensor:
        """Resolve a future, charging any blocked time to stall_seconds."""
        if fut.done():
            with self._stats_lock:
                self.stats.prefetch_hits += 1
            return fut.result()
        t0 = time.perf_counter()
        out = fut.result()
        dt = time.perf_counter() - t0
        with self._stats_lock:
            self.stats.stall_seconds += dt
            self.stats.prefetch_misses += 1
        return out

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        with self._handles_lock:
            for fh in self._handles:
                try:
                    fh.close()
                except OSError:
                    pass
            self._handles.clear()

    def __enter__(self) -> "AsyncReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
