"""Hard memory ceiling.

The library's one non-negotiable promise: peak resident bytes never exceed
`--mem-budget`, whatever the model size. The arena enforces it by blocking
allocation until an in-flight block retires, which back-pressures the
prefetcher instead of letting it run away and OOM.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """A single request is larger than the entire budget — unsatisfiable."""


@dataclass
class ArenaStats:
    peak_bytes: int = 0
    total_acquired: int = 0
    n_waits: int = 0
    wait_seconds: float = 0.0


class Arena:
    """Counting semaphore over bytes, with blocking acquire."""

    def __init__(self, budget_bytes: int) -> None:
        if budget_bytes <= 0:
            raise ValueError("budget must be positive")
        self.budget = budget_bytes
        self._used = 0
        self._lock = threading.Condition()
        self.stats = ArenaStats()

    @property
    def used(self) -> int:
        return self._used

    @property
    def available(self) -> int:
        return self.budget - self._used

    def acquire(self, nbytes: int, timeout: float | None = None) -> None:
        if nbytes > self.budget:
            raise BudgetExceeded(
                f"request of {nbytes / 1e6:.1f}MB exceeds the whole "
                f"{self.budget / 1e6:.1f}MB budget; raise --mem-budget or "
                f"lower --block-rows"
            )
        import time

        with self._lock:
            if self._used + nbytes > self.budget:
                self.stats.n_waits += 1
                t0 = time.perf_counter()
                ok = self._lock.wait_for(
                    lambda: self._used + nbytes <= self.budget, timeout
                )
                self.stats.wait_seconds += time.perf_counter() - t0
                if not ok:
                    raise TimeoutError(
                        f"arena starved waiting for {nbytes / 1e6:.1f}MB"
                    )
            self._used += nbytes
            self.stats.total_acquired += nbytes
            self.stats.peak_bytes = max(self.stats.peak_bytes, self._used)

    def release(self, nbytes: int) -> None:
        with self._lock:
            self._used = max(0, self._used - nbytes)
            self._lock.notify_all()

    def reset(self) -> None:
        with self._lock:
            self._used = 0
            self.stats = ArenaStats()
            self._lock.notify_all()

    def summary(self) -> str:
        return (
            f"peak {self.stats.peak_bytes / 1e6:.1f}MB / "
            f"budget {self.budget / 1e6:.1f}MB | "
            f"waits {self.stats.n_waits} ({self.stats.wait_seconds:.2f}s)"
        )


class Lease:
    """Context manager tying a byte reservation to a scope."""

    __slots__ = ("arena", "nbytes")

    def __init__(self, arena: Arena, nbytes: int) -> None:
        self.arena = arena
        self.nbytes = nbytes

    def __enter__(self) -> "Lease":
        self.arena.acquire(self.nbytes)
        return self

    def __exit__(self, *exc: object) -> None:
        self.arena.release(self.nbytes)
