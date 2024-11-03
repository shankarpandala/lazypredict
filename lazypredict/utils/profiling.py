# lazypredict/utils/profiling.py

import cProfile
import pstats
import io
from typing import Any, Dict, List, Optional, Tuple, Union, Callable


class Profiler:
    """
    Profiler class to profile code execution using cProfile.

    Attributes:
        profiler (cProfile.Profile): cProfile profiler instance.
    """

    def __init__(self):
        """
        Initializes the Profiler.
        """
        self.profiler = cProfile.Profile()

    def start(self):
        """
        Starts the profiler.
        """
        self.profiler.enable()

    def stop(self):
        """
        Stops the profiler.
        """
        self.profiler.disable()

    def print_stats(self, sort_by: str = "cumtime", lines: int = 20):
        """
        Prints the profiling statistics.

        Args:
            sort_by (str, optional): Sort key for the stats. Defaults to "cumtime".
            lines (int, optional): Number of lines to print. Defaults to 20.
        """
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(lines)
        print(s.getvalue())
