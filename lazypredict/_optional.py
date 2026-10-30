"""Optional dependencies."""

import warnings


def warn_if_broken(package: str, error: Exception) -> None:
    """Report why the optional dependency *package* could not be imported.

    A package that is not installed raises ``ImportError``; lazypredict then
    simply works without it. A package that is installed but fails to load
    raises something else, for example XGBoost or LightGBM on macOS without
    the OpenMP runtime (``brew install libomp``). That case is skipped as well,
    but with a warning, instead of making ``import lazypredict`` fail.
    """
    if isinstance(error, ImportError):
        return
    warnings.warn(
        f"{package} is installed but could not be imported "
        f"({type(error).__name__}: {error}); lazypredict continues without it.",
        RuntimeWarning,
        stacklevel=2,
    )
