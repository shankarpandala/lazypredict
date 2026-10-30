"""Per-model time limits."""

import signal
import threading
from contextlib import contextmanager
from typing import Iterator, Optional, Union

from lazypredict.exceptions import TimeoutException

# setitimer() overflows a little above 9.2e9 seconds; longer limits, including
# float("inf"), cannot fire anyway and are treated as "no preemptive limit".
_MAX_TIMER_SECONDS = 1e9


@contextmanager
def time_limit(seconds: Optional[Union[int, float]]) -> Iterator[bool]:
    """Raise :class:`TimeoutException` if the block runs longer than *seconds*.

    The limit is enforced with a real-time interval timer (``SIGALRM``), which
    is only possible on POSIX systems, in the main thread, and when no other
    code (e.g. ``pytest-timeout``) is already using that timer. Code running
    inside a C extension is interrupted once it returns control to Python.

    Yields
    ------
    bool
        True when the limit is enforced preemptively, False when it is not
        (no limit, unsupported platform or thread, or timer already in use);
        callers should then check the elapsed time themselves.
    """
    can_interrupt = (
        seconds is not None
        and 0 < seconds < _MAX_TIMER_SECONDS  # also False for NaN and infinity
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
        and signal.getitimer(signal.ITIMER_REAL)[0] == 0
    )
    if not can_interrupt:
        yield False
        return

    def _raise_timeout(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, float(seconds))  # type: ignore[arg-type]
        yield True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
