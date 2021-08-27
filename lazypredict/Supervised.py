"""
Supervised Models
"""
# Author: Shankar Rao Pandala <shankar.pandala@live.com>

import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import time
import sklearn
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
from catboost import CatBoostRegressor

import lightgbm
import sklearn.multioutput
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.gaussian_process


warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

removed_classifiers = [
    # ("CheckingClassifier", sklearn.utils._mocking.CheckingClassifier),
    ("ClassifierChain", sklearn.multioutput.ClassifierChain),
    ("ComplementNB", sklearn.naive_bayes.ComplementNB),
    (
        "GradientBoostingClassifier",
        sklearn.ensemble.GradientBoostingClassifier,
    ),
    (
        "GaussianProcessClassifier",
        sklearn.gaussian_process.GaussianProcessClassifier,
    ),
    (
        "HistGradientBoostingClassifier",
        sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier,
    ),
    ("MLPClassifier", sklearn.neural_network.MLPClassifier),
    ("LogisticRegressionCV", sklearn.linear_model.LogisticRegressionCV),
    ("MultiOutputClassifier", sklearn.multioutput.MultiOutputClassifier),
    ("MultinomialNB", sklearn.naive_bayes.MultinomialNB),
    ("OneVsOneClassifier", sklearn.multiclass.OneVsOneClassifier),
    ("OneVsRestClassifier", sklearn.multiclass.OneVsRestClassifier),
    ("OutputCodeClassifier", sklearn.multiclass.OutputCodeClassifier),
    (
        "RadiusNeighborsClassifier",
        sklearn.neighbors.RadiusNeighborsClassifier,
    ),
    ("VotingClassifier", sklearn.ensemble.VotingClassifier),
]
removed_regressors = [
    ("TheilSenRegressor", sklearn.linear_model.TheilSenRegressor),
    ("ARDRegression", sklearn.linear_model.ARDRegression),
    ("CCA", sklearn.cross_decomposition.CCA),
    ("IsotonicRegression", sklearn.isotonic.IsotonicRegression),
    ("StackingRegressor",sklearn.ensemble.StackingRegressor),
    ("MultiOutputRegressor", sklearn.multioutput.MultiOutputRegressor),
    ("MultiTaskElasticNet", sklearn.linear_model.MultiTaskElasticNet),
    ("MultiTaskElasticNetCV", sklearn.linear_model.MultiTaskElasticNetCV),
    ("MultiTaskLasso", sklearn.linear_model.MultiTaskLasso),
    ("MultiTaskLassoCV", sklearn.linear_model.MultiTaskLassoCV),
    ("PLSCanonical", sklearn.cross_decomposition.PLSCanonical),
    ("PLSRegression", sklearn.cross_decomposition.PLSRegression),
    ("RadiusNeighborsRegressor", sklearn.neighbors.RadiusNeighborsRegressor),
    ("RegressorChain", sklearn.multioutput.RegressorChain),
    ("VotingRegressor", sklearn.ensemble.VotingRegressor),
    # ("_SigmoidCalibration", sklearn.calibration._SigmoidCalibration),
]

CLASSIFIERS = [est for est in all_estimators() if
               (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))]
REGRESSORS = [est for est in all_estimators() if
              (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))]

REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
# REGRESSORS.append(('CatBoostRegressor',catboost.CatBoostRegressor))

CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))
# CLASSIFIERS.append(('CatBoostClassifier',catboost.CatBoostClassifier))

# REGRESSORS=[('Linear regression', sklearn.linear_model._base.LinearRegression),
#       ('AdaBoostRegressor', sklearn.ensemble._weight_boosting.AdaBoostRegressor),
#       ('Support vector machine', sklearn.svm._classes.SVR),
#       ('NuSVR', sklearn.svm._classes.NuSVR),
#       ('LinearSVR', sklearn.svm._classes.LinearSVR),
#       ('BaggingRegressor', sklearn.ensemble._bagging.BaggingRegressor),
#       ('Decision tree',sklearn.tree._classes.DecisionTreeRegressor),
#       ('Random forest', sklearn.ensemble._forest.RandomForestRegressor),
#       ('Gradient boosting Regressor', sklearn.ensemble._gb.GradientBoostingRegressor),
#       ('LightGBM', lightgbm.LGBMRegressor),
#       ('XGBoost', xgboost.XGBRegressor),
#       ('ExtraTreesRegressor', sklearn.ensemble._forest.ExtraTreesRegressor),
#       ('KNeighborsRegressor', sklearn.neighbors._regression.KNeighborsRegressor),
#       ('Lasso Regression',sklearn.linear_model.Lasso),
#       ('Ridge Regression', sklearn.linear_model.Ridge),
#       ('CatboostRegressor', CatBoostRegressor),
      
# ]


numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
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


# metrics function
def adjusted_rsquared(r2, n, p):
    return 1 - (1-r2) * ((n-1) / (n-p-1))

def bias(evaluation, simulation):
    if (len(evaluation) == len(simulation)):
        obs, sim = np.array(evaluation), np.array(simulation)
        bias = np.nansum(obs - sim) / len(obs)
        return float(bias)

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def pbias(evaluation, simulation):    
    if len(evaluation) == len(simulation):
        sim = np.array(simulation)
        obs = np.array(evaluation)
        return 100 * (float(np.nansum(sim - obs)) / float(np.nansum(obs)))
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def nashsutcliffe(evaluation, simulation):    
    if len(evaluation) == len(simulation):
        s, e = np.array(simulation), np.array(evaluation)
        # s,e=simulation,evaluation
        mean_observed = np.nanmean(e)
        # compute numerator and denominator
        numerator = np.nansum((e - s) ** 2)
        denominator = np.nansum((e - mean_observed)**2)
        # compute coefficient
        return 1 - (numerator / denominator)

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def lognashsutcliffe(evaluation, simulation, epsilon=0):
    
    if len(evaluation) == len(simulation):
        s, e = np.array(simulation)+epsilon, np.array(evaluation)+epsilon
        return float(1 - sum((np.log(s) - np.log(e))**2) / sum((np.log(e) - np.mean(np.log(e)))**2))
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def log_p(evaluation, simulation):
    scale = np.mean(evaluation) / 10
    if scale < .01:
        scale = .01
    if len(evaluation) == len(simulation):
        y = (np.array(evaluation) - np.array(simulation)) / scale
        normpdf = -y**2 / 2 - np.log(np.sqrt(2 * np.pi))
        return np.mean(normpdf)
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def correlationcoefficient(evaluation, simulation):    
    if len(evaluation) == len(simulation):
        correlation_coefficient = np.corrcoef(evaluation, simulation)[0, 1]
        return correlation_coefficient
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def rsquared(evaluation, simulation):
    
    if len(evaluation) == len(simulation):
        return (correlationcoefficient(evaluation, simulation)**2)
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def mse(evaluation, simulation):

    if len(evaluation) == len(simulation):
        obs, sim = np.array(evaluation), np.array(simulation)
        mse = np.nanmean((obs - sim)**2)
        return mse
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def rmse(evaluation, simulation):
    
    if len(evaluation) == len(simulation) > 0:
        return np.sqrt(mse(evaluation, simulation))
    else:
        logging.warning("evaluation and simulation lists do not have the same length.")
        return np.nan


def mae(evaluation, simulation):
    
    if len(evaluation) == len(simulation) > 0:
        obs, sim = np.array(evaluation), np.array(simulation)
        mae = np.mean(np.abs(sim - obs))
        return mae
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def rrmse(evaluation, simulation):
    """Relative Root Mean Squared Error"""

    if len(evaluation) == len(simulation):
        rrmse = rmse(evaluation, simulation) / np.mean(evaluation)
        return rrmse
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def agreementindex(evaluation, simulation):
    
    if len(evaluation) == len(simulation):
        simulation, evaluation = np.array(simulation), np.array(evaluation)
        Agreement_index = 1 - (np.sum((evaluation - simulation)**2)) / (np.sum(
            (np.abs(simulation - np.mean(evaluation)) + np.abs(evaluation - np.mean(evaluation)))**2))
        return Agreement_index
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def covariance(evaluation, simulation):
    
    if len(evaluation) == len(simulation):
        obs, sim = np.array(evaluation), np.array(simulation)
        obs_mean = np.mean(obs)
        sim_mean = np.mean(sim)
        covariance = np.mean((obs - obs_mean)*(sim - sim_mean))
        return covariance
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def decomposed_mse(evaluation, simulation):

    if len(evaluation) == len(simulation):
        e_std = np.std(evaluation)
        s_std = np.std(simulation)

        bias_squared = bias(evaluation, simulation)**2
        sdsd = (e_std - s_std)**2
        lcs = 2 * e_std * s_std * (1 - correlationcoefficient(evaluation, simulation))

        decomposed_mse = bias_squared + sdsd + lcs

        return decomposed_mse
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def kge(evaluation, simulation, return_all=False):
    
    if len(evaluation) == len(simulation):
        cc = np.corrcoef(evaluation, simulation)[0, 1]
        alpha = np.std(simulation) / np.std(evaluation)
        beta = np.sum(simulation) / np.sum(evaluation)
        kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        if return_all:
            return kge, cc, alpha, beta
        else:
            return kge
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan

def _spearmann_corr(x, y):
    
    col = [list(a) for a in zip(x, y)]
    xy = sorted(col, key=lambda x: x[0], reverse=False)
    # rang of x-value
    for i, row in enumerate(xy):
        row.append(i+1)

    a = sorted(xy, key=lambda x: x[1], reverse=False)
    # rang of y-value
    for i, row in enumerate(a):
        row.append(i+1)

    MW_rank_x = np.nanmean(np.array(a)[:,2])
    MW_rank_y = np.nanmean(np.array(a)[:,3])

    numerator = np.nansum([float((a[j][2]-MW_rank_x)*(a[j][3]-MW_rank_y)) for j in range(len(a))])
    denominator1 = np.sqrt(np.nansum([(a[j][2]-MW_rank_x)**2. for j in range(len(a))]))
    denominator2 = np.sqrt(np.nansum([(a[j][3]-MW_rank_x)**2. for j in range(len(a))]))
    return float(numerator/(denominator1*denominator2))

def kge_non_parametric(evaluation, simulation, return_all=False):
    
    if len(evaluation) == len(simulation):
        
        cc = _spearmann_corr(evaluation, simulation)

        fdc_sim = np.sort(simulation / (np.nanmean(simulation)*len(simulation)))
        fdc_obs = np.sort(evaluation / (np.nanmean(evaluation)*len(evaluation)))
        alpha = 1 - 0.5 * np.nanmean(np.abs(fdc_sim - fdc_obs))

        beta = np.mean(simulation) / np.mean(evaluation)
        kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        if return_all:
            return kge, cc, alpha, beta
        else:
            return kge
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan

def rsr(evaluation, simulation):
    if len(evaluation) == len(simulation):
        rsr = rmse(evaluation, simulation) / np.std(evaluation)
        return rsr
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def volume_error(evaluation, simulation):
    if len(evaluation) == len(simulation):
        ve = np.sum(simulation - evaluation) / np.sum(evaluation)
        return float(ve)
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan

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
        classifiers = "all"
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.classifiers = classifiers

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
                    full_name = (classifier.__class__.__name__, classifier)
                    temp_list.append(full_name)
                self.classifiers = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Classifier(s)")

        for name, model in tqdm(self.classifiers):
            start = time.time()
            try:
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
                names.append(name)
                Accuracy.append(accuracy)
                B_Accuracy.append(b_accuracy)
                ROC_AUC.append(roc_auc)
                F1.append(f1)
                TIME.append(time.time() - start)
                if self.custom_metric is not None:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)
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
            except Exception as exception:
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
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value 
            with key as name of models.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train,X_test,y_train,y_test)

        return self.models


def adjusted_rsquared(r2, n, p):
    return 1 - (1-r2) * ((n-1) / (n-p-1))


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

    >>> boston = datasets.load_boston()
    >>> X, y = shuffle(boston.data, boston.target, random_state=13)
    >>> X = X.astype(np.float32)

    >>> offset = int(X.shape[0] * 0.9)
    >>> X_train, y_train = X[:offset], y[:offset]
    >>> X_test, y_test = X[offset:], y[offset:]

    >>> reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    >>> models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    >>> model_dictionary = reg.provide_models(X_train, X_test, y_train, y_test)
    >>> models
    | Model                         | Adjusted R-Squared | R-Squared |  RMSE | Time Taken |
    |:------------------------------|-------------------:|----------:|------:|-----------:|
    | SVR                           |               0.83 |      0.88 |  2.62 |       0.01 |
    | BaggingRegressor              |               0.83 |      0.88 |  2.63 |       0.03 |
    | NuSVR                         |               0.82 |      0.86 |  2.76 |       0.03 |
    | RandomForestRegressor         |               0.81 |      0.86 |  2.78 |       0.21 |
    | XGBRegressor                  |               0.81 |      0.86 |  2.79 |       0.06 |
    | GradientBoostingRegressor     |               0.81 |      0.86 |  2.84 |       0.11 |
    | ExtraTreesRegressor           |               0.79 |      0.84 |  2.98 |       0.12 |
    | AdaBoostRegressor             |               0.78 |      0.83 |  3.04 |       0.07 |
    | HistGradientBoostingRegressor |               0.77 |      0.83 |  3.06 |       0.17 |
    | PoissonRegressor              |               0.77 |      0.83 |  3.11 |       0.01 |
    | LGBMRegressor                 |               0.77 |      0.83 |  3.11 |       0.07 |
    | KNeighborsRegressor           |               0.77 |      0.83 |  3.12 |       0.01 |
    | DecisionTreeRegressor         |               0.65 |      0.74 |  3.79 |       0.01 |
    | MLPRegressor                  |               0.65 |      0.74 |  3.80 |       1.63 |
    | HuberRegressor                |               0.64 |      0.74 |  3.84 |       0.01 |
    | GammaRegressor                |               0.64 |      0.73 |  3.88 |       0.01 |
    | LinearSVR                     |               0.62 |      0.72 |  3.96 |       0.01 |
    | RidgeCV                       |               0.62 |      0.72 |  3.97 |       0.01 |
    | BayesianRidge                 |               0.62 |      0.72 |  3.97 |       0.01 |
    | Ridge                         |               0.62 |      0.72 |  3.97 |       0.01 |
    | TransformedTargetRegressor    |               0.62 |      0.72 |  3.97 |       0.01 |
    | LinearRegression              |               0.62 |      0.72 |  3.97 |       0.01 |
    | ElasticNetCV                  |               0.62 |      0.72 |  3.98 |       0.04 |
    | LassoCV                       |               0.62 |      0.72 |  3.98 |       0.06 |
    | LassoLarsIC                   |               0.62 |      0.72 |  3.98 |       0.01 |
    | LassoLarsCV                   |               0.62 |      0.72 |  3.98 |       0.02 |
    | Lars                          |               0.61 |      0.72 |  3.99 |       0.01 |
    | LarsCV                        |               0.61 |      0.71 |  4.02 |       0.04 |
    | SGDRegressor                  |               0.60 |      0.70 |  4.07 |       0.01 |
    | TweedieRegressor              |               0.59 |      0.70 |  4.12 |       0.01 |
    | GeneralizedLinearRegressor    |               0.59 |      0.70 |  4.12 |       0.01 |
    | ElasticNet                    |               0.58 |      0.69 |  4.16 |       0.01 |
    | Lasso                         |               0.54 |      0.66 |  4.35 |       0.02 |
    | RANSACRegressor               |               0.53 |      0.65 |  4.41 |       0.04 |
    | OrthogonalMatchingPursuitCV   |               0.45 |      0.59 |  4.78 |       0.02 |
    | PassiveAggressiveRegressor    |               0.37 |      0.54 |  5.09 |       0.01 |
    | GaussianProcessRegressor      |               0.23 |      0.43 |  5.65 |       0.03 |
    | OrthogonalMatchingPursuit     |               0.16 |      0.38 |  5.89 |       0.01 |
    | ExtraTreeRegressor            |               0.08 |      0.32 |  6.17 |       0.01 |
    | DummyRegressor                |              -0.38 |     -0.02 |  7.56 |       0.01 |
    | LassoLars                     |              -0.38 |     -0.02 |  7.56 |       0.01 |
    | KernelRidge                   |             -11.50 |     -8.25 | 22.74 |       0.01 |
    """

 
    def __init__(
        self,
        verbose=False,
        ignore_warnings=True,
        custom_metric=None,
        predictions=True,
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
        actual=[]
        pred=[]
        R2 = []
        ADJR2 = []
        RMSE = []
        MSE2= []
        MAE2= []
        # WIN = []
        names = []
        TIME = []
        correlationcoeff=[]
        predictions = {}
        CUSTOM_METRIC = []
        decomp_mse= []
        rr_mse=[]
        co_var=[] 
        bi_as= []
        r_sr= []
        k_ge= []
        p_bias = []
        l_ogp= []
        volume_err= []
        agree_index= []
        log_nashsutcliffe= []
        nash_sutcliffe= [] 
        k=1
        


        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        transformers = []

        if len(numeric_features) > 0:
            transformers.append(("numeric", numeric_transformer, numeric_features))
        if len(categorical_features) > 0:
            categorical_low, categorical_high = get_card_split(
                X_train, categorical_features
            )

            if len(categorical_low) > 0:
                transformers.append(("categorical_low", categorical_transformer_low, categorical_low))
            if len(categorical_high) > 0:
                transformers.append(("categorical_high", categorical_transformer_high, categorical_high))

        preprocessor = ColumnTransformer(transformers=transformers)

        if self.regressors == "all":
            self.regressors = REGRESSORS
        else:
            try:
                temp_list = []
                for regressor in self.regressors:
                    full_name = (regressor.__class__.__name__, regressor)
                    temp_list.append(full_name)
                self.regressors = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Regressor(s)")

        for name, model in tqdm(self.regressors):
            start = time.time()
            try:
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


                r_squared = round( r2_score(y_test, y_pred), 3)
                mean_sq=round(mean_squared_error(y_test,y_pred) , 3)
                adj_rsquared = adjusted_rsquared(r_squared, X_test.shape[0], X_test.shape[1])
                rmse = round( np.sqrt(mean_squared_error(y_test, y_pred)), 3)
                mean_abs=mae(y_test,y_pred)
                covar= covariance(y_test, y_pred)
                r2mse=rrmse(y_test,y_pred)
                bia=bias(y_test,y_pred)
                corr=correlationcoefficient(y_test,y_pred)
                decompmse= decomposed_mse(y_test,y_pred)
                r_s_r=rsr(y_test,y_pred)
                k_g_e =kge(y_test,y_pred)
                p_b_ias=pbias(y_test,y_pred)
                l_o_gp = log_p(y_test,y_pred)
                vol_err= volume_error(y_test,y_pred)
                agreeindex=agreementindex(y_test, y_pred)
                lognashcl= lognashsutcliffe(y_test,y_pred)
                nascl= nashsutcliffe(y_test,y_pred)

                

                actual.append(y_test)
                pred.append(y_pred)
                names.append(name)
                R2.append(r_squared)
                ADJR2.append(adj_rsquared)
                RMSE.append(rmse)
                MSE2.append(mean_sq)
                MAE2.append(mean_abs)
                co_var.append(covar)
                rr_mse.append(r2mse)
                bi_as.append(bia)
                correlationcoeff.append(corr)
                decomp_mse.append(decompmse)
                r_sr.append(r_s_r)
                k_ge.append(k_g_e)
                p_bias.append(p_b_ias)
                l_ogp.append(l_o_gp)
                volume_err.append(vol_err)
                agree_index.append(agreeindex)
                log_nashsutcliffe.append(lognashcl)
                nash_sutcliffe.append(nascl)


                # aggrement_index.append(aggrement_index)
                
                dur= round( time.time() - start, 3)
                TIME.append(dur )

                if self.custom_metric:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)

                if self.verbose > 0:
                    scores_verbose = {
                        "Model": name,
                        # "Actual":y_test,
                        # "Predicted(U)":y_pred,
                        "R-Squared": r_squared,
                        "Adjusted R-Squared": adj_rsquared,
                        "RMSE": rmse,
                        "MSE":mean_sq,
                        "MAE":mean_abs,
                        "Correlation": corr,
                        "Covariance": covar,
                        "Decomposed MSE": decompmse,
                        "Realtive RMSE": r2mse,
                        "Bias":bia,
                        "RMSE S.D ratio":r_s_r,
                        "Pbias": p_b_ias,
                        "LogP": l_o_gp,
                        "Volume Error": vol_err,
                        "Agreement Index": agreeindex,
                        "LogNashsutCliffe": lognashcl,
                        "NashSutCliffe": nascl,
                        

                        "KGE":k_g_e,
                        "Time taken": round( time.time() - start, 3),
                    }

                    if self.custom_metric:
                        scores_verbose[self.custom_metric.__name__] = custom_metric

                    print(scores_verbose)
                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)

        scores = {
            "Model": names,
            # "Actual": actual,
            # "Predicted":pred,
            "Adjusted R-Squared": ADJR2,
            "R-Squared": R2,
            "RMSE": RMSE,
            "MSE":MSE2,
            "MAE":MAE2,
            "Covariance": co_var,
            "Relative RMSE": rr_mse,
            "Decomposed MSE": decomp_mse,
            "Bias": bi_as,
            "RMSE S.D ratio":r_sr,
            "Pbias": p_bias,
            "LogP": l_ogp,
            "Volume Error": volume_err,
            "KGE": k_ge,
            "Correlation coeff":correlationcoeff,
            "Agreement Index": agree_index,
            "NashSutCliffe": nash_sutcliffe,
            "LogNashsutCliffe": log_nashsutcliffe,
            

          
            # "Bias":bias,
            "Time Taken": TIME
        }

        if self.custom_metric:
            scores[self.custom_metric.__name__] = CUSTOM_METRIC

        scores = pd.DataFrame(scores)
        scores = scores.sort_values(by="Adjusted R-Squared", ascending=False).set_index("Model")

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
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value
            with key as name of models.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train,X_test,y_train,y_test)

        return self.models

Regression = LazyRegressor
Classification = LazyClassifier
