"""Injectable clocks so rate limits, backoff and breakers can be tested without real waiting."""
from __future__ import annotations

import asyncio
import time


class Clock:
    """Monotonic wall clock (seconds)."""

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class FakeClock(Clock):
    """Virtual clock: ``sleep`` advances time instantly (and yields to the event loop).

    Concurrent sleepers each advance the clock by their own duration, so this is exact for one
    sleeper at a time and an upper bound otherwise; tests should assert with that in mind.
    """

    def __init__(self, start: float = 0.0):
        self.t = float(start)
        self.slept: list = []

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += max(0.0, seconds)

    async def sleep(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        self.slept.append(seconds)
        self.t += seconds
        await asyncio.sleep(0)
