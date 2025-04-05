"""
Supervised Models
"""
# Author: Shankar Rao Pandala <shankar.pandala@live.com>

import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
try:
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        # We're in a Jupyter notebook or similar environment
        from tqdm.notebook import tqdm as notebook_tqdm
        use_notebook_tqdm = True
    else:
        use_notebook_tqdm = False
except:
    use_notebook_tqdm = False

import datetime
import time
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
import warnings
import xgboost

# import catboost
import lightgbm

# Import MLflow for model tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Check if MLflow tracking URI is set (like in Databricks 'databricks-uc')
def is_mlflow_tracking_enabled():
    """Checks if MLflow tracking is enabled via environment variable."""
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    return MLFLOW_AVAILABLE and tracking_uri is not None

# Initialize MLflow if tracking URI is set
def setup_mlflow():
    """Initialize MLflow if tracking URI is set through environment variable."""
    if is_mlflow_tracking_enabled():
        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.autolog()
        return True
    return False

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

removed_classifiers = [
    "ClassifierChain",
    "ComplementNB",
    "GradientBoostingClassifier",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    "MLPClassifier",
    "LogisticRegressionCV", 
    "MultiOutputClassifier", 
    "MultinomialNB", 
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
]

removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression", 
    "CCA", 
    "IsotonicRegression", 
    "StackingRegressor",
    "MultiOutputRegressor", 
    "MultiTaskElasticNet", 
    "MultiTaskElasticNetCV", 
    "MultiTaskLasso", 
    "MultiTaskLassoCV", 
    "PLSCanonical", 
    "PLSRegression", 
    "RadiusNeighborsRegressor", 
    "RegressorChain", 
    "VotingRegressor", 
]

CLASSIFIERS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
]

REGRESSORS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]

REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
# REGRESSORS.append(('CatBoostRegressor',catboost.CatBoostRegressor))

CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))
# CLASSIFIERS.append(('CatBoostClassifier',catboost.CatBoostClassifier))

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        # 'OrdianlEncoder' Raise a ValueError when encounters an unknown value. Check https://github.com/scikit-learn/scikit-learn/pull/13423
        ("encoding", OrdinalEncoder()),
    ]
)


# Helper function


def get_card_split(df, cols, n=11):
    """
    Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame from which the cardinality of the columns is calculated.
    cols : list-like
        Categorical columns to list
    n : int, optional (default=11)
        The value of 'n' will be used to split columns.
    Returns
    -------
    card_low : list-like
        Columns with cardinality < n
    card_high : list-like
        Columns with cardinality >= n
    """
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high


# Helper class for performing classification


class LazyClassifier:
    """
    This module helps in fitting to all the classification algorithms that are available in Scikit-learn
    Parameters
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.
    custom_metric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as dataframe.
    classifiers : list, optional (default="all")
        When function is provided, trains the chosen classifier(s).

    Examples
    --------
    >>> from lazypredict.Supervised import LazyClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_breast_cancer()
    >>> X = data.data
    >>> y= data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
    >>> clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    >>> models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    >>> model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test)
    >>> models
    | Model                          |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
    |:-------------------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
    | LinearSVC                      |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0150008 |
    | SGDClassifier                  |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0109992 |
    | MLPClassifier                  |   0.985965 |            0.986904 |  0.986904 |   0.985994 |    0.426     |
    | Perceptron                     |   0.985965 |            0.984797 |  0.984797 |   0.985965 |    0.0120046 |
    | LogisticRegression             |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.0200036 |
    | LogisticRegressionCV           |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.262997  |
    | SVC                            |   0.982456 |            0.979942 |  0.979942 |   0.982437 |    0.0140011 |
    | CalibratedClassifierCV         |   0.982456 |            0.975728 |  0.975728 |   0.982357 |    0.0350015 |
    | PassiveAggressiveClassifier    |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0130005 |
    | LabelPropagation               |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0429988 |
    | LabelSpreading                 |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0310006 |
    | RandomForestClassifier         |   0.97193  |            0.969594 |  0.969594 |   0.97193  |    0.033     |
    | GradientBoostingClassifier     |   0.97193  |            0.967486 |  0.967486 |   0.971869 |    0.166998  |
    | QuadraticDiscriminantAnalysis  |   0.964912 |            0.966206 |  0.966206 |   0.965052 |    0.0119994 |
    | HistGradientBoostingClassifier |   0.968421 |            0.964739 |  0.964739 |   0.968387 |    0.682003  |
    | RidgeClassifierCV              |   0.97193  |            0.963272 |  0.963272 |   0.971736 |    0.0130029 |
    | RidgeClassifier                |   0.968421 |            0.960525 |  0.960525 |   0.968242 |    0.0119977 |
    | AdaBoostClassifier             |   0.961404 |            0.959245 |  0.959245 |   0.961444 |    0.204998  |
    | ExtraTreesClassifier           |   0.961404 |            0.957138 |  0.957138 |   0.961362 |    0.0270066 |
    | KNeighborsClassifier           |   0.961404 |            0.95503  |  0.95503  |   0.961276 |    0.0560005 |
    | BaggingClassifier              |   0.947368 |            0.954577 |  0.954577 |   0.947882 |    0.0559971 |
    | BernoulliNB                    |   0.950877 |            0.951003 |  0.951003 |   0.951072 |    0.0169988 |
    | LinearDiscriminantAnalysis     |   0.961404 |            0.950816 |  0.950816 |   0.961089 |    0.0199995 |
    | GaussianNB                     |   0.954386 |            0.949536 |  0.949536 |   0.954337 |    0.0139935 |
    | NuSVC                          |   0.954386 |            0.943215 |  0.943215 |   0.954014 |    0.019989  |
    | DecisionTreeClassifier         |   0.936842 |            0.933693 |  0.933693 |   0.936971 |    0.0170023 |
    | NearestCentroid                |   0.947368 |            0.933506 |  0.933506 |   0.946801 |    0.0160074 |
    | ExtraTreeClassifier            |   0.922807 |            0.912168 |  0.912168 |   0.922462 |    0.0109999 |
    | CheckingClassifier             |   0.361404 |            0.5      |  0.5      |   0.191879 |    0.0170043 |
    | DummyClassifier                |   0.512281 |            0.489598 |  0.489598 |   0.518924 |    0.0119965 |
    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        classifiers="all",
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.classifiers = classifiers
        # Initialize MLflow if tracking URI is set
        self.mlflow_enabled = setup_mlflow()

    def fit(self, X_train, X_test, y_train, y_test):
        """Fit Classification algorithms to X_train and y_train, predict and score on X_test, y_test.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.
        """
        Accuracy = []
        B_Accuracy = []
        ROC_AUC = []
        F1 = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric is not None:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )

        if self.classifiers == "all":
            self.classifiers = CLASSIFIERS
        else:
            try:
                temp_list = []
                for classifier in self.classifiers:
                    full_name = (classifier.__name__, classifier)
                    temp_list.append(full_name)
                self.classifiers = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Classifier(s)")

        # Use notebook tqdm if in Jupyter environment
        progress_bar = notebook_tqdm if use_notebook_tqdm else tqdm
        for name, model in progress_bar(self.classifiers):
            start = time.time()
            try:
                # Start MLflow run for this specific model if MLflow is enabled
                mlflow_active_run = None
                if self.mlflow_enabled and MLFLOW_AVAILABLE:
                    mlflow_active_run = mlflow.start_run(run_name=f"LazyClassifier-{name}")
                    mlflow.log_param("model_name", name)
                    
                if "random_state" in model().get_params().keys():
                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("classifier", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("classifier", model())]
                    )

                pipe.fit(X_train, y_train)
                self.models[name] = pipe
                y_pred = pipe.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                try:
                    roc_auc = roc_auc_score(y_test, y_pred)
                except Exception as exception:
                    roc_auc = None
                    if self.ignore_warnings is False:
                        print("ROC AUC couldn't be calculated for " + name)
                        print(exception)
                
                # Log metrics to MLflow if enabled
                if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("balanced_accuracy", b_accuracy)
                    mlflow.log_metric("f1_score", f1)
                    if roc_auc is not None:
                        mlflow.log_metric("roc_auc", roc_auc)
                    mlflow.log_metric("training_time", time.time() - start)

                    # Log the model with signature
                    try:
                        signature = mlflow.models.infer_signature(X_train, pipe.predict(X_train))
                        mlflow.sklearn.log_model(pipe, f"{name}_model", signature=signature,
                                             registered_model_name=f"lazy_classifier_{name}")
                    except Exception as e:
                        if not self.ignore_warnings:
                            print(f"Failed to log model {name} to MLflow: {str(e)}")
                
                names.append(name)
                Accuracy.append(accuracy)
                B_Accuracy.append(b_accuracy)
                ROC_AUC.append(roc_auc)
                F1.append(f1)
                TIME.append(time.time() - start)
                if self.custom_metric is not None:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)
                    # Log custom metric to MLflow if enabled
                    if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                        mlflow.log_metric(self.custom_metric.__name__, custom_metric)
                
                if self.verbose > 0:
                    if self.custom_metric is not None:
                        print(
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,
                                "F1 Score": f1,
                                self.custom_metric.__name__: custom_metric,
                                "Time taken": time.time() - start,
                            }
                        )
                    else:
                        print(
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,
                                "F1 Score": f1,
                                "Time taken": time.time() - start,
                            }
                        )
                if self.predictions:
                    predictions[name] = y_pred
                    
                # End MLflow run for this model
                if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                    mlflow.end_run()
                    
            except Exception as exception:
                # End MLflow run if it was started but an error occurred
                if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                    mlflow.end_run()
                
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)
        if self.custom_metric is None:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    "Time Taken": TIME,
                }
            )
        else:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    self.custom_metric.__name__: CUSTOM_METRIC,
                    "Time Taken": TIME,
                }
            )
        scores = scores.sort_values(by="Balanced Accuracy", ascending=False).set_index(
            "Model"
        )

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value 
            with key as name of models.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models


def adjusted_rsquared(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


# Helper class for performing classification


class LazyRegressor:
    """
    This module helps in fitting regression models that are available in Scikit-learn
    Parameters
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.
    custom_metric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as dataframe.
    regressors : list, optional (default="all")
        When function is provided, trains the chosen regressor(s).

    Examples
    --------
    >>> from lazypredict.Supervised import LazyRegressor
    >>> from sklearn import datasets
    >>> from sklearn.utils import shuffle
    >>> import numpy as np

    >>> diabetes  = datasets.load_diabetes()
    >>> X, y = shuffle(diabetes.data, diabetes.target, random_state=13)
    >>> X = X.astype(np.float32)

    >>> offset = int(X.shape[0] * 0.9)
    >>> X_train, y_train = X[:offset], y[:offset]
    >>> X_test, y_test = X[offset:], y[offset:]

    >>> reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    >>> models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    >>> model_dictionary = reg.provide_models(X_train, X_test, y_train, y_test)
    >>> models
    | Model                         |   Adjusted R-Squared |   R-Squared |     RMSE |   Time Taken |
    |:------------------------------|---------------------:|------------:|---------:|-------------:|
    | ExtraTreesRegressor           |           0.378921   |  0.520076   |  54.2202 |   0.121466   |
    | OrthogonalMatchingPursuitCV   |           0.374947   |  0.517004   |  54.3934 |   0.0111742  |
    | Lasso                         |           0.373483   |  0.515873   |  54.457  |   0.00620174 |
    | LassoLars                     |           0.373474   |  0.515866   |  54.4575 |   0.0087235  |
    | LarsCV                        |           0.3715     |  0.514341   |  54.5432 |   0.0160234  |
    | LassoCV                       |           0.370413   |  0.513501   |  54.5903 |   0.0624897  |
    | PassiveAggressiveRegressor    |           0.366958   |  0.510831   |  54.7399 |   0.00689793 |
    | LassoLarsIC                   |           0.364984   |  0.509306   |  54.8252 |   0.0108321  |
    | SGDRegressor                  |           0.364307   |  0.508783   |  54.8544 |   0.0055306  |
    | RidgeCV                       |           0.363002   |  0.507774   |  54.9107 |   0.00728202 |
    | Ridge                         |           0.363002   |  0.507774   |  54.9107 |   0.00556874 |
    | BayesianRidge                 |           0.362296   |  0.507229   |  54.9411 |   0.0122972  |
    | LassoLarsCV                   |           0.361749   |  0.506806   |  54.9646 |   0.0175984  |
    | TransformedTargetRegressor    |           0.361749   |  0.506806   |  54.9646 |   0.00604773 |
    | LinearRegression              |           0.361749   |  0.506806   |  54.9646 |   0.00677514 |
    | Lars                          |           0.358828   |  0.504549   |  55.0903 |   0.00935149 |
    | ElasticNetCV                  |           0.356159   |  0.502486   |  55.2048 |   0.0478678  |
    | HuberRegressor                |           0.355251   |  0.501785   |  55.2437 |   0.0129263  |
    | RandomForestRegressor         |           0.349621   |  0.497434   |  55.4844 |   0.2331     |
    | AdaBoostRegressor             |           0.340416   |  0.490322   |  55.8757 |   0.0512381  |
    | LGBMRegressor                 |           0.339239   |  0.489412   |  55.9255 |   0.0396187  |
    | HistGradientBoostingRegressor |           0.335632   |  0.486625   |  56.0779 |   0.0897055  |
    | PoissonRegressor              |           0.323033   |  0.476889   |  56.6072 |   0.00953603 |
    | ElasticNet                    |           0.301755   |  0.460447   |  57.4899 |   0.00604224 |
    | KNeighborsRegressor           |           0.299855   |  0.458979   |  57.5681 |   0.00757337 |
    | OrthogonalMatchingPursuit     |           0.292421   |  0.453235   |  57.8729 |   0.00709486 |
    | BaggingRegressor              |           0.291213   |  0.452301   |  57.9223 |   0.0302746  |
    | GradientBoostingRegressor     |           0.247009   |  0.418143   |  59.7011 |   0.136803   |
    | TweedieRegressor              |           0.244215   |  0.415984   |  59.8118 |   0.00633955 |
    | XGBRegressor                  |           0.224263   |  0.400567   |  60.5961 |   0.339694   |
    | GammaRegressor                |           0.223895   |  0.400283   |  60.6105 |   0.0235181  |
    | RANSACRegressor               |           0.203535   |  0.38455    |  61.4004 |   0.0653253  |
    | LinearSVR                     |           0.116707   |  0.317455   |  64.6607 |   0.0077076  |
    | ExtraTreeRegressor            |           0.00201902 |  0.228833   |  68.7304 |   0.00626636 |
    | NuSVR                         |          -0.0667043  |  0.175728   |  71.0575 |   0.0143399  |
    | SVR                           |          -0.0964128  |  0.152772   |  72.0402 |   0.0114729  |
    | DummyRegressor                |          -0.297553   | -0.00265478 |  78.3701 |   0.00592971 |
    | DecisionTreeRegressor         |          -0.470263   | -0.136112   |  83.4229 |   0.00749898 |
    | GaussianProcessRegressor      |          -0.769174   | -0.367089   |  91.5109 |   0.0770502  |
    | MLPRegressor                  |          -1.86772    | -1.21597    | 116.508  |   0.235267   |
    | KernelRidge                   |          -5.03822    | -3.6659     | 169.061  |   0.0243919  |
    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        regressors="all",
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.regressors = regressors
        # Initialize MLflow if tracking URI is set
        self.mlflow_enabled = setup_mlflow()

    def fit(self, X_train, X_test, y_train, y_test):
        """Fit Regression algorithms to X_train and y_train, predict and score on X_test, y_test.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.
        """
        R2 = []
        ADJR2 = []
        RMSE = []
        # WIN = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )

        if self.regressors == "all":
            self.regressors = REGRESSORS
        else:
            try:
                temp_list = []
                for regressor in self.regressors:
                    full_name = (regressor.__name__, regressor)
                    temp_list.append(full_name)
                self.regressors = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Regressor(s)")

        # Use notebook tqdm if in Jupyter environment
        progress_bar = notebook_tqdm if use_notebook_tqdm else tqdm
        for name, model in progress_bar(self.regressors):
            start = time.time()
            try:
                # Start MLflow run for this specific model if MLflow is enabled
                mlflow_active_run = None
                if self.mlflow_enabled and MLFLOW_AVAILABLE:
                    mlflow_active_run = mlflow.start_run(run_name=f"LazyRegressor-{name}")
                    mlflow.log_param("model_name", name)
                    
                if "random_state" in model().get_params().keys():
                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("regressor", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("regressor", model())]
                    )

                pipe.fit(X_train, y_train)
                self.models[name] = pipe
                y_pred = pipe.predict(X_test)

                r_squared = r2_score(y_test, y_pred)
                adj_rsquared = adjusted_rsquared(
                    r_squared, X_test.shape[0], X_test.shape[1]
                )
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Log metrics to MLflow if enabled
                if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                    mlflow.log_metric("r_squared", r_squared)
                    mlflow.log_metric("adjusted_r_squared", adj_rsquared)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("training_time", time.time() - start)

                    # Log the model with signature
                    try:
                        signature = mlflow.models.infer_signature(X_train, pipe.predict(X_train))
                        mlflow.sklearn.log_model(pipe, f"{name}_model", signature=signature, 
                                              registered_model_name=f"lazy_regressor_{name}")
                    except Exception as e:
                        if not self.ignore_warnings:
                            print(f"Failed to log model {name} to MLflow: {str(e)}")

                names.append(name)
                R2.append(r_squared)
                ADJR2.append(adj_rsquared)
                RMSE.append(rmse)
                TIME.append(time.time() - start)

                if self.custom_metric:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)
                    # Log custom metric to MLflow if enabled
                    if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                        mlflow.log_metric(self.custom_metric.__name__, custom_metric)

                if self.verbose > 0:
                    scores_verbose = {
                        "Model": name,
                        "R-Squared": r_squared,
                        "Adjusted R-Squared": adj_rsquared,
                        "RMSE": rmse,
                        "Time taken": time.time() - start,
                    }

                    if self.custom_metric:
                        scores_verbose[self.custom_metric.__name__] = custom_metric

                    print(scores_verbose)
                if self.predictions:
                    predictions[name] = y_pred
                    
                # End MLflow run for this model
                if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                    mlflow.end_run()
                    
            except Exception as exception:
                # End MLflow run if it was started but an error occurred
                if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                    mlflow.end_run()
                    
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)

        scores = {
            "Model": names,
            "Adjusted R-Squared": ADJR2,
            "R-Squared": R2,
            "RMSE": RMSE,
            "Time Taken": TIME,
        }

        if self.custom_metric:
            scores[self.custom_metric.__name__] = CUSTOM_METRIC

        scores = pd.DataFrame(scores)
        scores = scores.sort_values(by="Adjusted R-Squared", ascending=False).set_index(
            "Model"
        )

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value 
            with key as name of models.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models


Regression = LazyRegressor
Classification = LazyClassifier
