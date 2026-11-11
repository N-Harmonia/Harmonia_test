from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class Stopwatch:
    splits: dict[str, float] = field(default_factory=dict)
    _running: dict[str, float] = field(default_factory=dict)

    def start(self, key: str) -> None:
        self._running[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        t0 = self._running.pop(key)
        elapsed = time.perf_counter() - t0
        self.splits[key] = self.splits.get(key, 0.0) + elapsed
        return elapsed

    @contextmanager
    def measure(self, key: str) -> Iterator[None]:
        self.start(key)
        try:
            yield
        finally:
            self.stop(key)

    def reset(self) -> None:
        self.splits.clear()
        self._running.clear()
