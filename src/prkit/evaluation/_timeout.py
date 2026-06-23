"""Thread-safe, Windows-safe bounded execution for ``simplify`` and friends.

PHYBench bounds ``simplify`` with a ``signal.SIGALRM`` decorator, which only works
on the main thread of the main interpreter — it raises if a scorer runs inside a
thread pool (the common eval-harness setup) and is unavailable on Windows. This
helper uses :mod:`concurrent.futures` instead.

A timed-out worker thread cannot be force-killed (SymPy's ``simplify`` is CPU-bound
pure Python), so it is *abandoned*: callers must treat :class:`SimplifyTimeout` as a
soft-degrade signal and must not retry in a tight loop.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FuturesTimeout
from typing import TypeVar

_T = TypeVar("_T")


class SimplifyTimeout(Exception):
    """Raised when a bounded computation exceeds its wall-clock deadline."""


def run_with_timeout(func: Callable[[], _T], *, timeout_s: float) -> _T:
    """Run ``func()`` with a wall-clock deadline.

    Args:
        func: a zero-argument callable (wrap arguments in a ``lambda``/closure).
        timeout_s: deadline in seconds.

    Returns:
        Whatever ``func()`` returns.

    Raises:
        SimplifyTimeout: if ``func`` does not finish within ``timeout_s``.

    Note:
        The executor is shut down with ``wait=False`` so a timed-out worker does
        not block return (``with ThreadPoolExecutor`` would join it on exit). The
        abandoned thread leaks until the underlying computation returns on its own.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(func)
    try:
        result = future.result(timeout=timeout_s)
    except _FuturesTimeout as exc:
        pool.shutdown(wait=False, cancel_futures=True)
        raise SimplifyTimeout(f"computation exceeded {timeout_s}s deadline") from exc
    pool.shutdown(wait=False)
    return result


__all__ = ["SimplifyTimeout", "run_with_timeout"]
