# lazypredict/utils/decorators.py

import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

def timeit(func: Callable) -> Callable:
    """
    Decorator that times the execution of a function.

    Args:
        func (Callable): Function to be timed.

    Returns:
        Callable: Wrapped function with timing.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")
        return result

    return wrapper


def profile(profiling_attr: str = 'profiling') -> Callable:
    """
    Decorator that profiles a function if the specified attribute is True.

    Args:
        profiling_attr (str, optional): Name of the attribute to check for profiling. Defaults to 'profiling'.

    Returns:
        Callable: Wrapped function with optional profiling.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            profiling_enabled = getattr(self, profiling_attr, False)
            if profiling_enabled:
                from .profiling import Profiler
                profiler = Profiler()
                profiler.start()
                result = func(self, *args, **kwargs)
                profiler.stop()
                profiler.print_stats()
            else:
                result = func(self, *args, **kwargs)
            return result

        return wrapper

    return decorator
