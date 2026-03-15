"""
Time Series Forecasting — LazyForecaster for rapid model benchmarking.

Provides a LazyForecaster class that trains multiple statistical, ML, and
deep-learning forecasting models to quickly identify which algorithms
perform best on a given time series.
"""
# Author: Shankar Rao Pandala <shankar.pandala@live.com>

import copy
import importlib.util
import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from lazypredict.config import (
    DEFAULT_N_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    DEFAULT_SORT_BY_FORECASTER,
    REMOVED_FORECASTERS,
    get_gpu_model_params,
    is_gpu_available,
)
from lazypredict.exceptions import InsufficientDataError
from lazypredict.integrations.mlflow import MLFLOW_AVAILABLE, setup_mlflow
from lazypredict.metrics import compute_forecast_metrics
from lazypredict.ts_preprocessing import (
    create_lag_features,
    detect_seasonal_period,
    recursive_forecast,
)

logger = logging.getLogger("lazypredict")

# ---------------------------------------------------------------------------
# Detect Jupyter notebook environment for tqdm
# ---------------------------------------------------------------------------
try:
    from IPython import get_ipython

    if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
        from tqdm.notebook import tqdm as notebook_tqdm

        _use_notebook_tqdm = True
    else:
        _use_notebook_tqdm = False
except Exception:
    _use_notebook_tqdm = False

# ---------------------------------------------------------------------------
# Optional MLflow
# ---------------------------------------------------------------------------
try:
    import mlflow as _mlflow
except ImportError:
    _mlflow = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optional dependencies — graceful degradation
# ---------------------------------------------------------------------------
try:
    import statsmodels  # noqa: F401

    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

try:
    import pmdarima  # noqa: F401

    _PMDARIMA_AVAILABLE = True
except ImportError:
    _PMDARIMA_AVAILABLE = False

try:
    _TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
except Exception:
    _TORCH_AVAILABLE = False

try:
    import xgboost  # noqa: F401

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    import lightgbm  # noqa: F401

    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

try:
    import catboost  # noqa: F401

    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

try:
    _TIMESFM_AVAILABLE = importlib.util.find_spec("timesfm") is not None
except Exception:
    _TIMESFM_AVAILABLE = False


# ============================================================================
# Forecaster Wrapper Interface
# ============================================================================


class ForecasterWrapper(ABC):
    """Abstract base class providing a uniform interface for all forecasting models.

    Every forecaster wrapper must implement :meth:`fit`, :meth:`predict`, and
    the :attr:`name` property so that :class:`LazyForecaster` can train and
    evaluate them interchangeably.
    """

    @abstractmethod
    def fit(
        self, y_train: np.ndarray, X_train: Optional[np.ndarray] = None
    ) -> None:
        """Fit the forecaster on training data.

        Parameters
        ----------
        y_train : np.ndarray
            1-D array of training observations in chronological order.
        X_train : np.ndarray or None, optional
            Exogenous feature matrix of shape ``(len(y_train), n_features)``.
        """

    @abstractmethod
    def predict(
        self, horizon: int, X_test: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forecast ``horizon`` steps into the future.

        Parameters
        ----------
        horizon : int
            Number of future time steps to forecast.
        X_test : np.ndarray or None, optional
            Exogenous features for the forecast period, shape
            ``(horizon, n_features)``.

        Returns
        -------
        np.ndarray
            1-D array of length ``horizon`` with point forecasts.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name used as index in result tables."""


# ============================================================================
# Statistical Model Wrappers (always available)
# ============================================================================


class NaiveForecaster(ForecasterWrapper):
    """Naive baseline that predicts the last observed value for all future steps.

    This is the simplest possible forecaster and serves as a lower-bound
    benchmark.  Exogenous features are ignored.
    """

    def fit(self, y_train, X_train=None):
        self._last_value = y_train[-1]

    def predict(self, horizon, X_test=None):
        return np.full(horizon, self._last_value)

    @property
    def name(self):
        return "Naive"


class SeasonalNaiveForecaster(ForecasterWrapper):
    """Seasonal naive baseline that repeats the last complete seasonal cycle.

    Parameters
    ----------
    seasonal_period : int, optional (default=1)
        Length of one seasonal cycle.  When set to 1 this behaves identically
        to :class:`NaiveForecaster`.
    """

    def __init__(self, seasonal_period: int = 1):
        self.seasonal_period = max(seasonal_period, 1)

    def fit(self, y_train, X_train=None):
        sp = min(self.seasonal_period, len(y_train))
        self._last_season = y_train[-sp:]

    def predict(self, horizon, X_test=None):
        repeats = (horizon // len(self._last_season)) + 1
        return np.tile(self._last_season, repeats)[:horizon]

    @property
    def name(self):
        return "SeasonalNaive"


# ============================================================================
# Statistical Model Wrappers (require statsmodels)
# ============================================================================


class SimpleExpSmoothingForecaster(ForecasterWrapper):
    """Simple Exponential Smoothing (no trend, no seasonality).

    Uses ``statsmodels.tsa.holtwinters.SimpleExpSmoothing`` with optimised
    smoothing parameters.  Suitable for series without trend or seasonality.
    Requires ``statsmodels``.
    """

    def fit(self, y_train, X_train=None):
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        self._model = SimpleExpSmoothing(y_train, initialization_method="estimated").fit(
            optimized=True
        )

    def predict(self, horizon, X_test=None):
        return np.asarray(self._model.forecast(horizon), dtype=float)

    @property
    def name(self):
        return "SimpleExpSmoothing"


class HoltForecaster(ForecasterWrapper):
    """Holt's linear trend method (double exponential smoothing).

    Captures level and trend components but ignores seasonality.
    Requires ``statsmodels``.
    """

    def fit(self, y_train, X_train=None):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self._model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)

    def predict(self, horizon, X_test=None):
        return np.asarray(self._model.forecast(horizon), dtype=float)

    @property
    def name(self):
        return "Holt"


class HoltWintersForecaster(ForecasterWrapper):
    """Holt-Winters triple exponential smoothing with additive or multiplicative seasonality.

    Parameters
    ----------
    seasonal : str, optional (default="add")
        Type of seasonal component: ``"add"`` or ``"mul"``.
    seasonal_periods : int or None, optional (default=None)
        Number of observations per seasonal cycle (e.g. 12 for monthly data
        with yearly seasonality).  Must be >= 2.
    label_suffix : str, optional (default="")
        Suffix appended to the model name for display.

    Requires ``statsmodels``.
    """

    def __init__(
        self,
        seasonal: str = "add",
        seasonal_periods: Optional[int] = None,
        label_suffix: str = "",
    ):
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._label_suffix = label_suffix

    def fit(self, y_train, X_train=None):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        sp = self.seasonal_periods
        if sp is None or sp < 2:
            sp = 2  # minimum for seasonal models
        if len(y_train) < 2 * sp:
            raise ValueError(
                f"Need at least 2 * seasonal_periods ({2 * sp}) observations, "
                f"got {len(y_train)}"
            )
        self._model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal=self.seasonal,
            seasonal_periods=sp,
            initialization_method="estimated",
        ).fit(optimized=True)

    def predict(self, horizon, X_test=None):
        return np.asarray(self._model.forecast(horizon), dtype=float)

    @property
    def name(self):
        return f"HoltWinters_{self.seasonal.capitalize()}{self._label_suffix}"


class ThetaForecaster(ForecasterWrapper):
    """Theta method from ``statsmodels``.

    The Theta method decomposes the series using a modified theta-line
    and produces forecasts by extrapolating with simple exponential smoothing.

    Parameters
    ----------
    seasonal_period : int or None, optional (default=None)
        Seasonal period passed to ``ThetaModel``.  ``None`` defaults to 1
        (no seasonality).

    Requires ``statsmodels``.
    """

    def __init__(self, seasonal_period: Optional[int] = None):
        self.seasonal_period = seasonal_period

    def fit(self, y_train, X_train=None):
        from statsmodels.tsa.forecasting.theta import ThetaModel

        period = self.seasonal_period or 1
        self._model = ThetaModel(y_train, period=period).fit()

    def predict(self, horizon, X_test=None):
        return np.asarray(self._model.forecast(horizon), dtype=float)

    @property
    def name(self):
        return "Theta"


class SARIMAXForecaster(ForecasterWrapper):
    """Seasonal ARIMA with eXogenous regressors (SARIMAX).

    Uses a default order of ``(1,1,1)`` with seasonal order
    ``(1,1,1,sp)`` when a seasonal period is detected.  Supports
    exogenous features via ``X_train`` / ``X_test``.

    Parameters
    ----------
    seasonal_period : int or None, optional (default=None)
        Seasonal period.  When <= 1 the seasonal component is disabled.

    Requires ``statsmodels``.
    """

    def __init__(self, seasonal_period: Optional[int] = None):
        self.seasonal_period = seasonal_period

    def fit(self, y_train, X_train=None):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        sp = self.seasonal_period or 1
        seasonal_order = (1, 1, 1, sp) if sp > 1 else (0, 0, 0, 0)
        self._model = SARIMAX(
            y_train,
            exog=X_train,
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    def predict(self, horizon, X_test=None):
        return np.asarray(
            self._model.forecast(steps=horizon, exog=X_test), dtype=float
        )

    @property
    def name(self):
        return "SARIMAX"


# ============================================================================
# Statistical Model Wrappers (require pmdarima)
# ============================================================================


class AutoARIMAForecaster(ForecasterWrapper):
    """Auto-ARIMA via ``pmdarima`` with automatic (p, d, q) parameter tuning.

    Performs a stepwise search over ARIMA orders to minimise information
    criteria.  Supports exogenous features.

    Requires ``pmdarima``.
    """

    def fit(self, y_train, X_train=None):
        from pmdarima import auto_arima

        self._model = auto_arima(
            y_train,
            exogenous=X_train,
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True,
        )

    def predict(self, horizon, X_test=None):
        return np.asarray(
            self._model.predict(n_periods=horizon, exogenous=X_test), dtype=float
        )

    @property
    def name(self):
        return "AutoARIMA"


# ============================================================================
# ML Model Wrapper (sklearn regressors with lag features)
# ============================================================================


class MLForecaster(ForecasterWrapper):
    """Wraps any scikit-learn regressor for time series by engineering lag and rolling features.

    The training series is transformed into a tabular supervised-learning
    problem using :func:`~lazypredict.ts_preprocessing.create_lag_features`.
    Multi-step forecasts are produced via recursive (autoregressive)
    prediction with :func:`~lazypredict.ts_preprocessing.recursive_forecast`.

    Parameters
    ----------
    estimator_class : type
        An sklearn-compatible regressor class (not an instance).
    model_name : str
        Display name used in result tables.
    n_lags : int, optional (default=10)
        Number of lag features to create.
    n_rolling : tuple of int, optional (default=(3, 7))
        Window sizes for rolling mean/std features.
    random_state : int, optional (default=42)
        Seed passed to the estimator if it accepts ``random_state``.
    """

    def __init__(
        self,
        estimator_class,
        model_name: str,
        n_lags: int = 10,
        n_rolling: Tuple[int, ...] = (3, 7),
        random_state: int = 42,
        use_gpu: bool = False,
    ):
        self.estimator_class = estimator_class
        self._model_name = model_name
        self.n_lags = n_lags
        self.n_rolling = n_rolling
        self.random_state = random_state
        self.use_gpu = use_gpu

    def fit(self, y_train, X_train=None):
        X_feat, y_feat = create_lag_features(
            y_train, self.n_lags, self.n_rolling, X_exog=X_train
        )
        params: dict = get_gpu_model_params(self.estimator_class, self.use_gpu)
        try:
            if "random_state" in self.estimator_class().get_params():
                params["random_state"] = self.random_state
        except Exception:
            pass
        # Suppress verbose output for boosting libraries by default
        module = getattr(self.estimator_class, "__module__", "") or ""
        if "catboost" in module:
            params.setdefault("verbose", 0)
        if "lightgbm" in module:
            params.setdefault("verbose", -1)
            params.setdefault("verbosity", -1)
        if "xgboost" in module:
            params.setdefault("verbosity", 0)
        self._estimator = self.estimator_class(**params)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_feat)
        self._estimator.fit(X_scaled, y_feat)
        self._y_train = y_train

    def predict(self, horizon, X_test=None):
        return recursive_forecast(
            self._estimator,
            self._scaler,
            self._y_train,
            horizon,
            self.n_lags,
            self.n_rolling,
            X_exog=X_test,
        )

    @property
    def name(self):
        return self._model_name


# ============================================================================
# Deep Learning Wrappers (require torch)
# ============================================================================


class _TorchRNNForecaster(ForecasterWrapper):
    """Base class for single-layer LSTM / GRU forecasters using PyTorch.

    Subclasses only need to set ``_rnn_type`` to ``"LSTM"`` or ``"GRU"``.
    Training uses early stopping with a patience of 5 epochs.

    Parameters
    ----------
    n_lags : int, optional (default=10)
        Number of lag features.
    n_rolling : tuple of int, optional (default=(3, 7))
        Rolling-window sizes for mean/std features.
    hidden_size : int, optional (default=64)
        Number of hidden units in the RNN layer.
    n_epochs : int, optional (default=50)
        Maximum training epochs.
    batch_size : int, optional (default=32)
        Mini-batch size for training.
    learning_rate : float, optional (default=1e-3)
        Adam optimiser learning rate.
    random_state : int, optional (default=42)
        Seed for reproducibility.

    Requires ``torch``.
    """

    _rnn_type: str = "LSTM"

    def __init__(
        self,
        n_lags: int = 10,
        n_rolling: Tuple[int, ...] = (3, 7),
        hidden_size: int = 64,
        n_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        random_state: int = 42,
        use_gpu: bool = False,
    ):
        self.n_lags = n_lags
        self.n_rolling = n_rolling
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.use_gpu = use_gpu

    def _get_device(self):
        """Determine the torch device to use."""
        import torch as _torch
        if self.use_gpu and _torch.cuda.is_available():
            return _torch.device("cuda")
        return _torch.device("cpu")

    def fit(self, y_train, X_train=None):
        import torch as _torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        _torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        device = self._get_device()

        X_feat, y_feat = create_lag_features(
            y_train, self.n_lags, self.n_rolling, X_exog=X_train
        )
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_feat)

        n_features = X_scaled.shape[1]
        X_tensor = _torch.tensor(
            X_scaled.reshape(-1, 1, n_features), dtype=_torch.float32
        ).to(device)
        y_tensor = _torch.tensor(
            y_feat.reshape(-1, 1), dtype=_torch.float32
        ).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        # Build model
        rnn_class = nn.LSTM if self._rnn_type == "LSTM" else nn.GRU
        self._rnn = rnn_class(
            input_size=n_features, hidden_size=self.hidden_size,
            num_layers=1, batch_first=True,
        ).to(device)
        self._linear = nn.Linear(self.hidden_size, 1).to(device)
        self._device = device
        params = list(self._rnn.parameters()) + list(self._linear.parameters())
        optimizer = _torch.optim.Adam(params, lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self._rnn.train()
        self._linear.train()
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                output, _ = self._rnn(X_batch)
                pred = self._linear(output[:, -1, :])
                loss = loss_fn(pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / max(len(loader), 1)
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break

        self._rnn.eval()
        self._linear.eval()
        self._y_train = y_train
        self._n_features = n_features

    def predict(self, horizon, X_test=None):
        import torch as _torch

        device = self._device
        predictions: List[float] = []
        history = list(self._y_train)
        n_rolling = self.n_rolling

        if X_test is not None:
            X_test = np.asarray(X_test)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)

        with _torch.no_grad():
            for step in range(horizon):
                features: List[float] = []
                for lag in range(1, self.n_lags + 1):
                    features.append(history[-lag])
                for window in n_rolling:
                    recent = history[-window:]
                    features.append(float(np.mean(recent)))
                    features.append(
                        float(np.std(recent)) if len(recent) > 1 else 0.0
                    )
                features.append(
                    history[-1] - history[-2] if len(history) >= 2 else 0.0
                )
                if X_test is not None and step < len(X_test):
                    features.extend(X_test[step].tolist())

                X_step = np.array(features).reshape(1, -1)
                X_step = self._scaler.transform(X_step)
                X_tensor = _torch.tensor(
                    X_step.reshape(1, 1, -1), dtype=_torch.float32
                ).to(device)
                output, _ = self._rnn(X_tensor)
                pred = self._linear(output[:, -1, :]).item()
                predictions.append(pred)
                history.append(pred)

        return np.array(predictions)

    @property
    def name(self):
        return f"{self._rnn_type}_TS"


class LSTMForecaster(_TorchRNNForecaster):
    """Single-layer LSTM forecaster.

    See :class:`_TorchRNNForecaster` for parameter details.
    Requires ``torch``.
    """

    _rnn_type = "LSTM"


class GRUForecaster(_TorchRNNForecaster):
    """Single-layer GRU forecaster.

    See :class:`_TorchRNNForecaster` for parameter details.
    Requires ``torch``.
    """

    _rnn_type = "GRU"


# ============================================================================
# Pretrained Foundation Model Wrapper (require timesfm)
# ============================================================================


class TimesFMForecaster(ForecasterWrapper):
    """Google TimesFM 2.5 zero-shot pretrained foundation model for forecasting.

    TimesFM is a 200M-parameter transformer pre-trained on a large corpus of
    real and synthetic time series.  It performs zero-shot inference—no task-
    specific training is needed.  Exogenous features are **not** supported and
    will be silently ignored.

    When ``use_gpu=True`` and CUDA is available, the model is placed on GPU
    for faster inference.

    Parameters
    ----------
    use_gpu : bool, optional (default=False)
        Place the model on a CUDA device when available.
    model_path : str or None, optional (default=None)
        Path to a local directory containing the pre-downloaded TimesFM
        model weights.  When ``None`` (default), the model is downloaded
        from Hugging Face (``google/timesfm-2.5-200m-pytorch``).
        Use this when you are offline or behind a firewall.

    Requires ``timesfm`` and ``torch`` (Python 3.10-3.11 only).
    """

    def __init__(self, use_gpu: bool = False, model_path: Optional[str] = None):
        self.use_gpu = use_gpu
        self.model_path = model_path

    def fit(self, y_train, X_train=None):
        import torch as _torch
        import timesfm as _timesfm

        _torch.set_float32_matmul_precision("high")

        # Determine device: GPU if requested and available
        backend = "gpu" if (self.use_gpu and _torch.cuda.is_available()) else "cpu"
        device = _torch.device("cuda" if backend == "gpu" else "cpu")

        repo_or_path = self.model_path or "google/timesfm-2.5-200m-pytorch"

        # If model_path is a local directory containing model.safetensors,
        # load directly via checkpoint (works offline without HF Hub).
        safetensors_file = (
            os.path.join(repo_or_path, "model.safetensors")
            if self.model_path and os.path.isdir(self.model_path)
            else None
        )

        if safetensors_file and os.path.isfile(safetensors_file):
            prev_offline = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                self._model = _timesfm.TimesFM_2p5_200M_torch()
                self._model.load_checkpoint(safetensors_file)
            finally:
                if prev_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev_offline
        else:
            # Try loading from HF cache first, then fall back to download
            kwargs = {"torch_device": device}
            try:
                self._model = _timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    repo_or_path, local_files_only=True, **kwargs,
                )
            except Exception:
                self._model = _timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    repo_or_path, **kwargs,
                )

        self._model.compile(
            _timesfm.ForecastConfig(
                max_context=min(len(y_train), 1024),
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self._y_train = np.asarray(y_train, dtype=float)
        if X_train is not None:
            logger.info(
                "TimesFM does not support exogenous variables; ignoring X_train"
            )

    def predict(self, horizon, X_test=None):
        point_forecast, _ = self._model.forecast(
            horizon=horizon, inputs=[self._y_train]
        )
        return np.asarray(point_forecast[0][:horizon], dtype=float)

    @property
    def name(self):
        return "TimesFM"


# ============================================================================
# Model Registry
# ============================================================================


def _build_forecaster_list(
    n_lags: int,
    n_rolling: Tuple[int, ...],
    random_state: int,
    seasonal_period: Optional[int],
    use_gpu: bool = False,
    foundation_model_path: Optional[str] = None,
) -> List[Tuple[str, ForecasterWrapper]]:
    """Build the default list of all available forecasters.

    Parameters
    ----------
    n_lags : int
        Number of lag features for ML / DL models.
    n_rolling : tuple of int
        Rolling-window sizes for ML / DL feature engineering.
    random_state : int
        Seed for reproducible ML / DL models.
    seasonal_period : int or None
        Detected or user-specified seasonal period.
    use_gpu : bool, optional (default=False)
        When True, enables GPU acceleration for models that support it.
    foundation_model_path : str or None, optional (default=None)
        Local path to pre-downloaded foundation model weights (e.g. TimesFM).
        When ``None``, models are downloaded from Hugging Face.

    Returns
    -------
    list of (str, ForecasterWrapper)
        Tuples of ``(model_name, wrapper_instance)``.
    """
    forecasters: List[Tuple[str, ForecasterWrapper]] = []

    # -- Baselines (always available) -----------------------------------------
    forecasters.append(("Naive", NaiveForecaster()))
    sp = seasonal_period or 1
    forecasters.append(("SeasonalNaive", SeasonalNaiveForecaster(seasonal_period=sp)))

    # -- statsmodels ----------------------------------------------------------
    if _STATSMODELS_AVAILABLE:
        forecasters.append(("SimpleExpSmoothing", SimpleExpSmoothingForecaster()))
        forecasters.append(("Holt", HoltForecaster()))
        if seasonal_period and seasonal_period >= 2:
            forecasters.append((
                "HoltWinters_Add",
                HoltWintersForecaster(
                    seasonal="add", seasonal_periods=seasonal_period
                ),
            ))
            forecasters.append((
                "HoltWinters_Mul",
                HoltWintersForecaster(
                    seasonal="mul", seasonal_periods=seasonal_period
                ),
            ))
        forecasters.append(("Theta", ThetaForecaster(seasonal_period=seasonal_period)))
        forecasters.append((
            "SARIMAX",
            SARIMAXForecaster(seasonal_period=seasonal_period),
        ))
    else:
        logger.info(
            "statsmodels not installed — ETS/Theta/SARIMAX models unavailable. "
            "Install with: pip install statsmodels"
        )

    # -- pmdarima -------------------------------------------------------------
    if _PMDARIMA_AVAILABLE:
        forecasters.append(("AutoARIMA", AutoARIMAForecaster()))
    else:
        logger.info(
            "pmdarima not installed — AutoARIMA unavailable. "
            "Install with: pip install pmdarima"
        )

    # -- ML models (always available via sklearn) -----------------------------
    ml_models: List[Tuple[str, Any]] = [
        ("LinearRegression_TS", LinearRegression),
        ("Ridge_TS", Ridge),
        ("Lasso_TS", Lasso),
        ("ElasticNet_TS", ElasticNet),
        ("KNeighborsRegressor_TS", KNeighborsRegressor),
        ("DecisionTreeRegressor_TS", DecisionTreeRegressor),
        ("RandomForestRegressor_TS", RandomForestRegressor),
        ("GradientBoostingRegressor_TS", GradientBoostingRegressor),
        ("AdaBoostRegressor_TS", AdaBoostRegressor),
        ("ExtraTreesRegressor_TS", ExtraTreesRegressor),
        ("BaggingRegressor_TS", BaggingRegressor),
        ("SVR_TS", SVR),
    ]
    if _XGBOOST_AVAILABLE:
        ml_models.append(("XGBRegressor_TS", xgboost.XGBRegressor))
    if _LIGHTGBM_AVAILABLE:
        ml_models.append(("LGBMRegressor_TS", lightgbm.LGBMRegressor))
    if _CATBOOST_AVAILABLE:
        ml_models.append(("CatBoostRegressor_TS", catboost.CatBoostRegressor))

    for ml_name, ml_class in ml_models:
        forecasters.append((
            ml_name,
            MLForecaster(
                estimator_class=ml_class,
                model_name=ml_name,
                n_lags=n_lags,
                n_rolling=n_rolling,
                random_state=random_state,
                use_gpu=use_gpu,
            ),
        ))

    # -- Deep learning (requires torch) ---------------------------------------
    if _TORCH_AVAILABLE:
        forecasters.append((
            "LSTM_TS",
            LSTMForecaster(
                n_lags=n_lags, n_rolling=n_rolling, random_state=random_state,
                use_gpu=use_gpu,
            ),
        ))
        forecasters.append((
            "GRU_TS",
            GRUForecaster(
                n_lags=n_lags, n_rolling=n_rolling, random_state=random_state,
                use_gpu=use_gpu,
            ),
        ))
    else:
        logger.info(
            "torch not installed — LSTM/GRU models unavailable. "
            "Install with: pip install torch"
        )

    # -- Pretrained foundation models (requires timesfm) ----------------------
    if _TIMESFM_AVAILABLE:
        forecasters.append((
            "TimesFM",
            TimesFMForecaster(use_gpu=use_gpu, model_path=foundation_model_path),
        ))
    else:
        logger.info(
            "timesfm not installed — TimesFM model unavailable. "
            "Install with: pip install timesfm[torch]"
        )

    # Filter out removed forecasters
    forecasters = [
        (n, w) for n, w in forecasters if n not in REMOVED_FORECASTERS
    ]

    return forecasters


# ============================================================================
# LazyForecaster
# ============================================================================


class LazyForecaster:
    """Fit multiple time series forecasting models and benchmark them.

    Runs statistical, machine-learning, deep-learning, and pretrained
    foundation models on a time series and returns a ranked DataFrame of
    metrics so you can quickly see which approach works best.

    Parameters
    ----------
    verbose : int, optional (default=0)
        Controls progress-bar visibility and per-model metric logging.
    ignore_warnings : bool, optional (default=True)
        When True, model-level exceptions are silently stored in
        ``self.errors`` and the loop continues.
    custom_metric : callable or None, optional (default=None)
        Additional metric function ``f(y_true, y_pred) -> float``.
    predictions : bool, optional (default=False)
        When True, ``fit()`` returns a second DataFrame of predictions.
    random_state : int, optional (default=42)
        Seed for ML and deep-learning models.
    forecasters : str or list, optional (default="all")
        ``"all"`` to run every available model, or a list of model names
        (strings) to select a subset.
    cv : int or None, optional (default=None)
        Number of ``TimeSeriesSplit`` folds for cross-validation.
    timeout : int, float, or None, optional (default=None)
        Maximum training time in seconds per model.
    n_lags : int, optional (default=10)
        Number of lag features for ML/DL models.
    n_rolling : tuple of int, optional (default=(3, 7))
        Rolling-window sizes for feature engineering.
    seasonal_period : int or None, optional (default=None)
        Seasonal period.  ``None`` triggers auto-detection via ACF.
    sort_by : str, optional (default="RMSE")
        Metric column to sort results by.
    n_jobs : int, optional (default=-1)
        Parallel jobs for cross-validation.
    max_models : int or None, optional (default=None)
        Limit the number of models to train.
    progress_callback : callable or None, optional (default=None)
        Called after each model as ``f(name, current, total, metrics)``.
    use_gpu : bool, optional (default=False)
        When True, enables GPU acceleration for models that support it
        (e.g., XGBoost, LightGBM, LSTM, GRU). Falls back to CPU if CUDA
        is unavailable.
    foundation_model_path : str or None, optional (default=None)
        Local filesystem path to pre-downloaded foundation model weights
        (e.g. TimesFM).  Use this when you are offline, behind a firewall,
        or in an air-gapped environment.  When ``None`` (default), the
        model is downloaded from Hugging Face automatically.
    tune : bool, optional (default=False)
        When True, tunes the top-k forecasters after initial benchmarking
        using Optuna with temporal cross-validation.
    tune_top_k : int, optional (default=5)
        Number of top models to tune.
    tune_trials : int, optional (default=30)
        Number of Optuna trials per model.
    tune_timeout : int, float, or None, optional (default=None)
        Maximum seconds per model tuning.
    tune_metric : str, optional (default='RMSE')
        Metric to optimize: ``'RMSE'``, ``'MAE'``, ``'MAPE'``, ``'SMAPE'``, ``'MASE'``.
    tune_seasonal : bool, optional (default=False)
        When True, also search over seasonal_period values during tuning.
    horizon_strategy : str, optional (default='recursive')
        Multi-step forecast strategy for ML models:
        ``'recursive'`` (default), ``'direct'``, or ``'multi_output'``.

    Attributes
    ----------
    models : dict
        Fitted ``ForecasterWrapper`` objects keyed by model name.
    errors : dict
        Exceptions from models that failed, keyed by model name.
    tuned_scores_ : pd.DataFrame or None
        Tuning results if ``tune=True`` was set.
    """

    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[Callable] = None,
        predictions: bool = False,
        random_state: int = 42,
        forecasters: Union[str, List[str]] = "all",
        cv: Optional[int] = None,
        timeout: Optional[Union[int, float]] = None,
        n_lags: int = DEFAULT_N_LAGS,
        n_rolling: Tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
        seasonal_period: Optional[int] = None,
        sort_by: str = DEFAULT_SORT_BY_FORECASTER,
        n_jobs: int = -1,
        max_models: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        use_gpu: bool = False,
        foundation_model_path: Optional[str] = None,
        # Tuning parameters
        tune: bool = False,
        tune_top_k: int = 5,
        tune_trials: int = 30,
        tune_timeout: Optional[Union[int, float]] = None,
        tune_metric: str = "RMSE",
        tune_seasonal: bool = False,
        # Horizon strategy
        horizon_strategy: str = "recursive",
    ):
        # Validate
        if cv is not None and (not isinstance(cv, int) or cv < 2):
            raise ValueError(f"cv must be an integer >= 2, got {cv!r}")
        if timeout is not None and (
            not isinstance(timeout, (int, float)) or timeout <= 0
        ):
            raise ValueError(
                f"timeout must be a positive number, got {timeout!r}"
            )
        if custom_metric is not None and not callable(custom_metric):
            raise TypeError(
                f"custom_metric must be callable, got {type(custom_metric)}"
            )
        if max_models is not None and (
            not isinstance(max_models, int) or max_models < 1
        ):
            raise ValueError(
                f"max_models must be a positive integer, got {max_models!r}"
            )
        if tune_metric.upper() not in {"RMSE", "MAE", "MAPE", "SMAPE", "MASE"}:
            raise ValueError(
                f"tune_metric must be one of RMSE, MAE, MAPE, SMAPE, MASE, got {tune_metric!r}"
            )
        if horizon_strategy not in ("recursive", "direct", "multi_output"):
            raise ValueError(
                f"horizon_strategy must be 'recursive', 'direct', or 'multi_output', got {horizon_strategy!r}"
            )

        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.random_state = random_state
        self.forecasters = forecasters
        self.cv = cv
        self.timeout = timeout
        self.n_lags = n_lags
        self.n_rolling = n_rolling
        self.seasonal_period = seasonal_period
        self.sort_by = sort_by
        self.n_jobs = n_jobs
        self.max_models = max_models
        self.progress_callback = progress_callback
        self.use_gpu = use_gpu
        self.foundation_model_path = foundation_model_path
        self.tune = tune
        self.tune_top_k = tune_top_k
        self.tune_trials = tune_trials
        self.tune_timeout = tune_timeout
        self.tune_metric = tune_metric
        self.tune_seasonal = tune_seasonal
        self.horizon_strategy = horizon_strategy
        self.tuned_scores_: Optional[pd.DataFrame] = None
        self.tuned_models_: Dict[str, ForecasterWrapper] = {}

        self.models: Dict[str, ForecasterWrapper] = {}
        self.errors: Dict[str, Exception] = {}
        self.mlflow_enabled = setup_mlflow()

        if self.use_gpu:
            if is_gpu_available():
                logger.info("GPU acceleration enabled. CUDA is available.")
            else:
                logger.warning(
                    "GPU requested but CUDA is not available. "
                    "Models that require GPU will fall back to CPU."
                )

    # --------------------------------------------------------------------- #
    # Public API                                                              #
    # --------------------------------------------------------------------- #

    def _validate_fit_inputs(self, y_train, y_test, X_train, X_test):
        """Validate and coerce fit inputs to numpy arrays."""
        y_train = np.asarray(y_train, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        if len(y_train) == 0:
            raise ValueError("y_train is empty")
        if len(y_test) == 0:
            raise ValueError("y_test is empty")
        if X_train is not None:
            X_train = np.asarray(X_train, dtype=float)
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
        if X_test is not None:
            X_test = np.asarray(X_test, dtype=float)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
        min_required = self.n_lags + max(self.n_rolling) + 1
        if len(y_train) < min_required:
            raise InsufficientDataError(
                f"y_train has {len(y_train)} observations but at least "
                f"{min_required} are required (n_lags={self.n_lags}, "
                f"max rolling window={max(self.n_rolling)})"
            )
        return y_train, y_test, X_train, X_test

    def _fit_single_model(
        self, fname, wrapper, y_train, y_test, X_train, X_test,
        horizon, seasonal_period, results, predictions_dict,
    ):
        """Fit, predict, and score a single forecaster model."""
        start = time.time()
        mlflow_active_run = None
        try:
            if self.mlflow_enabled and MLFLOW_AVAILABLE and _mlflow is not None:
                mlflow_active_run = _mlflow.start_run(
                    run_name=f"LazyForecaster-{fname}"
                )
                _mlflow.log_param("model_name", fname)

            wrapper_copy = copy.deepcopy(wrapper)
            wrapper_copy.fit(y_train, X_train)
            fit_time = time.time() - start

            if self.timeout and fit_time > self.timeout:
                logger.info(
                    "%s exceeded timeout (%.2fs > %ss), skipping...",
                    fname, fit_time, self.timeout,
                )
                self._end_mlflow_run(mlflow_active_run)
                return None

            y_pred = wrapper_copy.predict(horizon, X_test)
            self.models[fname] = wrapper_copy

            sp = seasonal_period or 1
            metrics = compute_forecast_metrics(
                y_test, y_pred, y_train, seasonal_period=sp
            )
            metrics["name"] = fname
            metrics["time"] = time.time() - start

            if self.cv:
                cv_metrics = self._run_ts_cv(
                    wrapper, y_train, X_train, fname, seasonal_period
                )
                metrics.update(cv_metrics)

            self._compute_custom_metric(metrics, y_test, y_pred, fname)
            self._log_mlflow_metrics(mlflow_active_run, metrics)

            results.append(metrics)
            if self.verbose > 0:
                self._log_verbose(fname, metrics)
            # Always store predictions internally for ensemble/plotting/diagnostics
            predictions_dict[fname] = y_pred

            self._end_mlflow_run(mlflow_active_run)
            return metrics

        except Exception as exc:
            self._end_mlflow_run(mlflow_active_run)
            self.errors[fname] = exc
            if not self.ignore_warnings:
                logger.warning("%s model failed: %s", fname, exc)
            return None

    def _compute_custom_metric(self, metrics, y_test, y_pred, fname):
        """Run user-supplied custom metric if configured."""
        if self.custom_metric is None:
            return
        try:
            metrics["custom_metric"] = self.custom_metric(y_test, y_pred)
        except Exception as custom_exc:
            metrics["custom_metric"] = None
            if not self.ignore_warnings:
                logger.warning(
                    "Custom metric failed for %s: %s", fname, custom_exc
                )

    @staticmethod
    def _end_mlflow_run(mlflow_active_run):
        """End an MLflow run if one is active."""
        if mlflow_active_run and _mlflow is not None:
            _mlflow.end_run()

    @staticmethod
    def _log_mlflow_metrics(mlflow_active_run, metrics):
        """Log all numeric metrics to MLflow if a run is active."""
        if not (mlflow_active_run and _mlflow is not None):
            return
        for key, val in metrics.items():
            if isinstance(val, (int, float)) and val is not None:
                _mlflow.log_metric(key, val)

    def fit(
        self,
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        X_train: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit forecasting models and evaluate on test data.

        Parameters
        ----------
        y_train : array-like
            Training time series (chronological order).
        y_test : array-like
            Held-out future values to forecast against.
        X_train : array-like or None
            Exogenous features for the training period.
        X_test : array-like or None
            Exogenous features for the forecast period.

        Returns
        -------
        scores : pd.DataFrame
            Metric table for every model, sorted by ``sort_by``.
        predictions : pd.DataFrame
            Per-model predictions (empty if ``self.predictions`` is False).
        """
        y_train, y_test, X_train, X_test = self._convert_and_validate_inputs(
            y_train, y_test, X_train, X_test
        )
        horizon = len(y_test)
        seasonal_period = self._resolve_seasonal_period(y_train)

        all_forecasters = self._resolve_forecasters(seasonal_period)

        results: List[Dict[str, Any]] = []
        predictions_dict: Dict[str, np.ndarray] = {}

        self._run_all_models(
            all_forecasters, y_train, y_test, X_train, X_test,
            horizon, seasonal_period, results, predictions_dict,
        )

        scores = self._build_scores_dataframe(results)
        self._store_fit_state(
            y_train, y_test, X_train, seasonal_period,
            all_forecasters, predictions_dict, scores,
        )
        self._auto_tune(scores, all_forecasters, y_train, X_train, seasonal_period)

        if self.predictions:
            return scores, pd.DataFrame.from_dict(predictions_dict)
        return scores, pd.DataFrame()

    def _convert_and_validate_inputs(self, y_train, y_test, X_train, X_test):
        from lazypredict.distributed import auto_convert_dataframe

        y_train = auto_convert_dataframe(y_train, "y_train")
        y_test = auto_convert_dataframe(y_test, "y_test")
        if X_train is not None:
            X_train = auto_convert_dataframe(X_train, "X_train")
        if X_test is not None:
            X_test = auto_convert_dataframe(X_test, "X_test")
        return self._validate_fit_inputs(y_train, y_test, X_train, X_test)

    def _resolve_seasonal_period(self, y_train):
        seasonal_period = self.seasonal_period
        if seasonal_period is None:
            seasonal_period = detect_seasonal_period(y_train)
            if seasonal_period is not None and self.verbose > 0:
                logger.info("Auto-detected seasonal period: %d", seasonal_period)
        return seasonal_period

    def _resolve_forecasters(self, seasonal_period):
        all_forecasters = _build_forecaster_list(
            n_lags=self.n_lags,
            n_rolling=self.n_rolling,
            random_state=self.random_state,
            seasonal_period=seasonal_period,
            use_gpu=self.use_gpu,
            foundation_model_path=self.foundation_model_path,
        )
        if self.forecasters != "all":
            selected_names = set(self.forecasters)
            all_forecasters = [
                (n, w) for n, w in all_forecasters if n in selected_names
            ]
        if self.max_models is not None:
            all_forecasters = all_forecasters[: self.max_models]
        return all_forecasters

    def _run_all_models(
        self, all_forecasters, y_train, y_test, X_train, X_test,
        horizon, seasonal_period, results, predictions_dict,
    ):
        progress_bar = notebook_tqdm if _use_notebook_tqdm else tqdm
        total = len(all_forecasters)
        with warnings.catch_warnings():
            if self.ignore_warnings:
                warnings.simplefilter("ignore")
            for idx, (fname, wrapper) in enumerate(
                progress_bar(all_forecasters, disable=(self.verbose == 0))
            ):
                metrics = self._fit_single_model(
                    fname, wrapper, y_train, y_test, X_train, X_test,
                    horizon, seasonal_period, results, predictions_dict,
                )
                if self.progress_callback is not None:
                    self.progress_callback(fname, idx + 1, total, metrics)

    def _store_fit_state(
        self, y_train, y_test, X_train, seasonal_period,
        all_forecasters, predictions_dict, scores,
    ):
        self._last_y_train = y_train
        self._last_y_test = y_test
        self._last_X_train = X_train
        self._last_seasonal_period = seasonal_period
        self._last_all_forecasters = all_forecasters
        self._last_predictions = predictions_dict
        self._last_scores = scores

    def _auto_tune(self, scores, all_forecasters, y_train, X_train, seasonal_period):
        if not self.tune or len(scores) == 0:
            return
        from lazypredict.ts_tuning import tune_top_k_forecasters

        logger.info(
            "Tuning top %d forecasters with Optuna (%d trials, metric=%s)...",
            self.tune_top_k, self.tune_trials, self.tune_metric,
        )
        self.tuned_scores_, self.tuned_models_ = tune_top_k_forecasters(
            scores_df=scores,
            models=self.models,
            all_wrappers=all_forecasters,
            y_train=y_train,
            X_train=X_train,
            seasonal_period=seasonal_period,
            top_k=self.tune_top_k,
            tune_metric=self.tune_metric,
            cv=max(self.cv or 3, 3),
            n_trials=self.tune_trials,
            timeout=self.tune_timeout,
            random_state=self.random_state,
            tune_seasonal=self.tune_seasonal,
            refit=True,
        )

    def ensemble(
        self,
        method: str = "weighted_average",
        y_true: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Ensemble predictions from fitted models.

        Parameters
        ----------
        method : str, optional (default='weighted_average')
            Ensemble method: ``'simple_average'``, ``'weighted_average'``,
            or ``'stacking'``.
        y_true : np.ndarray or None
            True values (required for 'stacking' and 'weighted_average').

        Returns
        -------
        np.ndarray
            Ensembled predictions.
        """
        if not self._last_predictions:
            raise ValueError(
                "No predictions available. Call fit() with predictions=True first."
            )

        preds = self._last_predictions
        handlers = {
            "simple_average": self._ensemble_simple,
            "weighted_average": self._ensemble_weighted,
            "stacking": self._ensemble_stacking,
        }
        handler = handlers.get(method)
        if handler is None:
            raise ValueError(
                f"Unknown method: {method}. Use {', '.join(repr(k) for k in handlers)}."
            )
        return handler(preds, y_true)

    def _ensemble_simple(self, preds, y_true):
        from lazypredict.ensemble import ensemble_simple_average
        return ensemble_simple_average(preds)

    def _ensemble_weighted(self, preds, y_true):
        from lazypredict.ensemble import ensemble_weighted_average
        scores = {name: 1.0 for name in preds if name in self.models}
        if hasattr(self, "_last_scores_dict"):
            for name, s in self._last_scores_dict.items():
                if name in preds:
                    scores[name] = s
        return ensemble_weighted_average(preds, scores)

    def _ensemble_stacking(self, preds, y_true):
        from lazypredict.ensemble import ensemble_stacking
        if y_true is None:
            raise ValueError("y_true is required for stacking ensemble.")
        return ensemble_stacking(preds, y_true)

    def plot_results(
        self,
        y_train: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        plot_type: str = "forecast",
        **kwargs,
    ):
        """Generate forecast visualizations.

        Parameters
        ----------
        y_train : np.ndarray or None
            Training series. Uses stored values from last ``fit()`` if None.
        y_test : np.ndarray or None
            Test series. Uses stored values from last ``fit()`` if None.
        plot_type : str
            One of ``'forecast'``, ``'comparison'``, ``'residuals'``,
            ``'errors'``, ``'heatmap'``.
        **kwargs
            Passed to the underlying plot function (e.g. ``figsize``,
            ``metric``, ``model_name``, ``top_k``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        y_train = y_train if y_train is not None else getattr(self, "_last_y_train", None)
        y_test = y_test if y_test is not None else getattr(self, "_last_y_test", None)
        predictions = getattr(self, "_last_predictions", {})
        scores = getattr(self, "_last_scores", None)

        handlers = {
            "forecast": self._plot_forecast,
            "comparison": self._plot_comparison,
            "residuals": self._plot_residuals,
            "errors": self._plot_errors,
            "heatmap": self._plot_heatmap,
        }
        handler = handlers.get(plot_type)
        if handler is None:
            raise ValueError(
                f"Unknown plot_type: {plot_type!r}. "
                f"Use {', '.join(repr(k) for k in handlers)}."
            )
        return handler(y_train, y_test, predictions, scores, **kwargs)

    def _plot_forecast(self, y_train, y_test, predictions, scores, **kwargs):
        from lazypredict.ts_visualization import plot_forecast

        if y_train is None or y_test is None:
            raise ValueError("y_train and y_test required. Call fit() first.")
        if not predictions:
            raise ValueError("No predictions available. Call fit() first.")
        return plot_forecast(
            y_train, y_test, predictions,
            title=kwargs.get("title"),
            figsize=kwargs.get("figsize", (12, 5)),
            ax=kwargs.get("ax"),
        )

    def _plot_comparison(self, y_train, y_test, predictions, scores, **kwargs):
        from lazypredict.ts_visualization import plot_model_comparison

        if scores is None or scores.empty:
            raise ValueError("No scores available. Call fit() first.")
        return plot_model_comparison(
            scores,
            metric=kwargs.get("metric", "RMSE"),
            top_k=kwargs.get("top_k", 10),
            figsize=kwargs.get("figsize", (10, 6)),
            ax=kwargs.get("ax"),
        )

    def _plot_residuals(self, y_train, y_test, predictions, scores, **kwargs):
        from lazypredict.ts_visualization import plot_residuals

        if y_test is None:
            raise ValueError("y_test required. Call fit() first.")
        model_name = kwargs.get("model_name")
        if model_name:
            if model_name not in predictions:
                raise ValueError(
                    f"Model '{model_name}' not found in predictions. "
                    f"Available: {list(predictions.keys())}"
                )
            y_pred = predictions[model_name]
        else:
            model_name = next(iter(predictions))
            y_pred = predictions[model_name]
        return plot_residuals(
            y_test, y_pred, model_name=model_name,
            seasonal_period=kwargs.get("seasonal_period", getattr(self, "_last_seasonal_period", 1) or 1),
            figsize=kwargs.get("figsize", (12, 10)),
        )

    def _plot_errors(self, y_train, y_test, predictions, scores, **kwargs):
        from lazypredict.ts_visualization import plot_error_distribution

        if y_test is None:
            raise ValueError("y_test required. Call fit() first.")
        if not predictions:
            raise ValueError("No predictions available. Call fit() first.")
        return plot_error_distribution(
            y_test, predictions,
            figsize=kwargs.get("figsize", (10, 6)),
            ax=kwargs.get("ax"),
        )

    def _plot_heatmap(self, y_train, y_test, predictions, scores, **kwargs):
        from lazypredict.ts_visualization import plot_metrics_heatmap

        if scores is None or scores.empty:
            raise ValueError("No scores available. Call fit() first.")
        return plot_metrics_heatmap(
            scores,
            metrics=kwargs.get("metrics"),
            figsize=kwargs.get("figsize", (10, 8)),
            ax=kwargs.get("ax"),
        )

    def diagnose(
        self,
        model_name: Optional[str] = None,
        y_test: Optional[np.ndarray] = None,
    ):
        """Run residual diagnostics on fitted models.

        Parameters
        ----------
        model_name : str or None
            Specific model name. If None, runs diagnostics for all models
            and returns a DataFrame.
        y_test : np.ndarray or None
            Test values. Uses stored values from last ``fit()`` if None.

        Returns
        -------
        dict or pd.DataFrame
            Single model returns a diagnostic dict; multiple models
            returns a DataFrame with one row per model.
        """
        from lazypredict.ts_diagnostics import compare_diagnostics, residual_diagnostics

        y_test = y_test if y_test is not None else getattr(self, "_last_y_test", None)
        if y_test is None:
            raise ValueError("y_test required. Call fit() first or pass y_test.")

        predictions = getattr(self, "_last_predictions", {})
        if not predictions:
            raise ValueError("No predictions available. Call fit() first.")

        y_train = getattr(self, "_last_y_train", None)
        sp = getattr(self, "_last_seasonal_period", 1) or 1

        if model_name is not None:
            if model_name not in predictions:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    f"Available: {list(predictions.keys())}"
                )
            return residual_diagnostics(
                y_test, predictions[model_name],
                y_train=y_train, seasonal_period=sp,
            )

        return compare_diagnostics(
            y_test, predictions,
            y_train=y_train, seasonal_period=sp,
        )

    def provide_models(
        self,
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        X_train: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> Dict[str, ForecasterWrapper]:
        """Return all fitted forecaster wrappers.

        Calls :meth:`fit` automatically if no models have been fitted yet.

        Parameters
        ----------
        y_train : array-like
            Training time series.
        y_test : array-like
            Test time series (needed only if ``fit`` has not been called).
        X_train : array-like or None, optional
            Exogenous features for the training period.
        X_test : array-like or None, optional
            Exogenous features for the forecast period.

        Returns
        -------
        dict
            Mapping of model name to fitted :class:`ForecasterWrapper`.
        """
        if len(self.models) == 0:
            self.fit(y_train, y_test, X_train, X_test)
        return self.models

    def predict(
        self,
        y_history: Union[pd.Series, np.ndarray],
        horizon: int,
        model_name: Optional[str] = None,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Produce forecasts from previously fitted models.

        Each model is re-fit on ``y_history`` before predicting so that the
        most recent observations are used.

        Parameters
        ----------
        y_history : array-like
            Historical time series to condition the forecast on.
        horizon : int
            Number of future time steps to forecast.
        model_name : str or None, optional
            If given, only this model is used and a single ``np.ndarray`` is
            returned.  Otherwise all fitted models are used and a ``dict``
            mapping model names to arrays is returned.
        X_test : array-like or None, optional
            Exogenous features for the forecast period.

        Returns
        -------
        np.ndarray or dict
            A single forecast array when ``model_name`` is specified, or a
            ``{name: np.ndarray}`` dict for all models.

        Raises
        ------
        ValueError
            If no models have been fitted or ``model_name`` is not found.
        """
        if len(self.models) == 0:
            raise ValueError(
                "No models fitted yet. Call fit() first."
            )
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    f"Available: {list(self.models.keys())}"
                )
            wrapper = self.models[model_name]
            wrapper.fit(np.asarray(y_history, dtype=float))
            return wrapper.predict(horizon, X_test)
        result = {}
        for name, wrapper in self.models.items():
            wrapper.fit(np.asarray(y_history, dtype=float))
            result[name] = wrapper.predict(horizon, X_test)
        return result

    def save_models(self, path: str) -> None:
        """Save all fitted models to disk using ``joblib``.

        Parameters
        ----------
        path : str
            Directory path.  Created if it does not exist.

        Raises
        ------
        ValueError
            If no models have been fitted yet.
        """
        import os

        import joblib

        if len(self.models) == 0:
            raise ValueError("No models fitted yet. Call fit() first.")
        os.makedirs(path, exist_ok=True)
        for name, wrapper in self.models.items():
            filepath = os.path.join(path, f"{name}.joblib")
            joblib.dump(wrapper, filepath)
            logger.info("Saved %s to %s", name, filepath)

    def load_models(self, path: str) -> Dict[str, ForecasterWrapper]:
        """Load previously saved models from disk.

        Parameters
        ----------
        path : str
            Directory containing ``.joblib`` files written by
            :meth:`save_models`.

        Returns
        -------
        dict
            Mapping of model name to :class:`ForecasterWrapper`.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        import os

        import joblib

        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        for filename in os.listdir(path):
            if filename.endswith(".joblib"):
                name = filename[:-7]
                filepath = os.path.join(path, filename)
                self.models[name] = joblib.load(filepath)
                logger.info("Loaded %s from %s", name, filepath)
        return self.models

    # --------------------------------------------------------------------- #
    # Internal helpers                                                        #
    # --------------------------------------------------------------------- #

    def _run_ts_cv(
        self,
        wrapper: ForecasterWrapper,
        y_train: np.ndarray,
        X_train: Optional[np.ndarray],
        fname: str,
        seasonal_period: Optional[int],
    ) -> Dict[str, Optional[float]]:
        """Run time series cross-validation with an expanding window.

        Uses ``sklearn.model_selection.TimeSeriesSplit`` so that training
        data always precedes validation data chronologically.

        Returns a dict of ``{metric_name: value}`` with mean and standard
        deviation for each fold.
        """
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=self.cv)
        cv_scores: Dict[str, List[float]] = {
            m: [] for m in ["mae", "rmse", "mape", "smape", "mase"]
        }

        for train_idx, val_idx in tscv.split(y_train):
            y_cv_train = y_train[train_idx]
            y_cv_val = y_train[val_idx]
            X_cv_train = X_train[train_idx] if X_train is not None else None
            X_cv_val = X_train[val_idx] if X_train is not None else None

            try:
                w = copy.deepcopy(wrapper)
                w.fit(y_cv_train, X_cv_train)
                y_cv_pred = w.predict(len(val_idx), X_cv_val)
                sp = seasonal_period or 1
                fold_metrics = compute_forecast_metrics(
                    y_cv_val, y_cv_pred, y_cv_train, seasonal_period=sp
                )
                for metric_name in cv_scores:
                    cv_scores[metric_name].append(fold_metrics[metric_name])
            except Exception:
                for metric_name in cv_scores:
                    cv_scores[metric_name].append(np.nan)

        result: Dict[str, Optional[float]] = {}
        for metric_name, values in cv_scores.items():
            arr = np.array(values, dtype=float)
            label = metric_name.upper()
            result[f"{label} CV Mean"] = (
                float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else None
            )
            result[f"{label} CV Std"] = (
                float(np.nanstd(arr)) if not np.all(np.isnan(arr)) else None
            )
        return result

    @staticmethod
    def _cv_column_names() -> List[str]:
        """Return the ordered list of CV column names for the scores table."""
        return [
            "MAE CV Mean", "MAE CV Std",
            "RMSE CV Mean", "RMSE CV Std",
            "MAPE CV Mean", "MAPE CV Std",
            "SMAPE CV Mean", "SMAPE CV Std",
            "MASE CV Mean", "MASE CV Std",
        ]

    def _build_scores_dataframe(
        self, results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Assemble per-model metric dicts into a sorted ``DataFrame``."""
        if not results:
            return pd.DataFrame()

        rows = []
        for r in results:
            row: Dict[str, Any] = {
                "Model": r["name"],
                "MAE": r["mae"],
                "RMSE": r["rmse"],
                "MAPE": r["mape"],
                "SMAPE": r["smape"],
                "MASE": r["mase"],
                "R-Squared": r["r_squared"],
            }
            # CV columns
            for col in self._cv_column_names():
                if col in r:
                    row[col] = r[col]
            # Custom metric
            if self.custom_metric is not None:
                row[self.custom_metric.__name__] = r.get("custom_metric")
            row["Time Taken"] = r["time"]
            rows.append(row)

        scores = pd.DataFrame(rows)

        # Sort — lower is better for error metrics, higher for R-Squared
        ascending = self.sort_by != "R-Squared"
        if self.sort_by in scores.columns:
            scores = scores.sort_values(
                by=self.sort_by, ascending=ascending
            )
        scores = scores.set_index("Model")
        return scores

    @staticmethod
    def _log_verbose(name: str, metrics: Dict[str, Any]) -> None:
        """Log a single model's metrics to the ``lazypredict`` logger."""
        logger.info(
            "Model: %-35s MAE: %.4f  RMSE: %.4f  MAPE: %.2f  MASE: %.4f  Time: %.2fs",
            name,
            metrics.get("mae", 0),
            metrics.get("rmse", 0),
            metrics.get("mape", 0),
            metrics.get("mase", 0),
            metrics.get("time", 0),
        )
