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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.testing import all_estimators
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, r2_score, mean_squared_error
import warnings
libnames = ['xgboost', 'catboost', 'lightgbm']
for libname in libnames:
    try:
        lib = __import__(libname)
    except:
        print(sys.exc_info())
    else:
        globals()[libname] = lib
warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: '%.2f' % x)

CLASSIFIERS = [est for est in all_estimators(
) if issubclass(est[1], ClassifierMixin)]
REGRESSORS = [est for est in all_estimators(
) if issubclass(est[1], RegressorMixin)]

removed_classifiers = [('ClassifierChain', sklearn.multioutput.ClassifierChain),
 ('ComplementNB', sklearn.naive_bayes.ComplementNB),
 ('GradientBoostingClassifier',
  sklearn.ensemble.gradient_boosting.GradientBoostingClassifier),
 ('GaussianProcessClassifier',sklearn.gaussian_process.gpc.GaussianProcessClassifier),
 ('HistGradientBoostingClassifier',
  sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier),
 ('MLPClassifier', sklearn.neural_network.multilayer_perceptron.MLPClassifier),
 ('LogisticRegressionCV', sklearn.linear_model.logistic.LogisticRegressionCV),
 ('MultiOutputClassifier', sklearn.multioutput.MultiOutputClassifier),
 ('MultinomialNB', sklearn.naive_bayes.MultinomialNB),
 ('OneVsOneClassifier', sklearn.multiclass.OneVsOneClassifier),
 ('OneVsRestClassifier', sklearn.multiclass.OneVsRestClassifier),
 ('OutputCodeClassifier', sklearn.multiclass.OutputCodeClassifier),
 ('RadiusNeighborsClassifier',
  sklearn.neighbors.classification.RadiusNeighborsClassifier),
 ('VotingClassifier', sklearn.ensemble.voting.VotingClassifier)]

removed_regressors = [('TheilSenRegressor', sklearn.linear_model.theil_sen.TheilSenRegressor),
 ('ARDRegression', sklearn.linear_model.ARDRegression),
 ('CCA', sklearn.cross_decomposition.CCA),
 ('IsotonicRegression', sklearn.isotonic.IsotonicRegression),
 ('MultiOutputRegressor', sklearn.multioutput.MultiOutputRegressor),
 ('MultiTaskElasticNet',
  sklearn.linear_model.MultiTaskElasticNet),
 ('MultiTaskElasticNetCV',
  sklearn.linear_model.MultiTaskElasticNetCV),
 ('MultiTaskLasso', sklearn.linear_model.MultiTaskLasso),
 ('MultiTaskLassoCV',
  sklearn.linear_model.MultiTaskLassoCV),
 ('PLSCanonical', sklearn.cross_decomposition.PLSCanonical),
 ('PLSRegression', sklearn.cross_decomposition.PLSRegression),
 ('RadiusNeighborsRegressor',
  sklearn.neighbors.RadiusNeighborsRegressor),
 ('RegressorChain', sklearn.multioutput.RegressorChain),
 ('VotingRegressor', sklearn.ensemble.VotingRegressor),
 ('_SigmoidCalibration', sklearn.calibration._SigmoidCalibration)]

for i in removed_regressors:
    REGRESSORS.pop(REGRESSORS.index(i))
    
for i in removed_classifiers:
    CLASSIFIERS.pop(CLASSIFIERS.index(i))

REGRESSORS.append(('XGBRegressor', xgboost.XGBRegressor))
REGRESSORS.append(('LGBMRegressor',lightgbm.LGBMRegressor))
REGRESSORS.append(('CatBoostRegressor',catboost.CatBoostRegressor))
    
CLASSIFIERS.append(('XGBClassifier',xgboost.XGBClassifier))
CLASSIFIERS.append(('LGBMClassifier',lightgbm.LGBMClassifier))
CLASSIFIERS.append(('CatBoostClassifier',catboost.CatBoostClassifier))

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoding', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

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

    def __init__(self, verbose=0, ignore_warnings=True, custom_metric = None, predictions = False):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions

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
        
        if self.custom_metric != None:
            CUSTOM_METRIC = []
            
        if type(X_train) is np.ndarray:
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(
            include=['int64', 'float64', 'int32', 'float32']).columns
        categorical_features = X_train.select_dtypes(
            include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features)
            ])

        for name, model in tqdm(CLASSIFIERS):
            start = time.time()
            try:
                pipe = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', model())
                ])
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                try:
                    roc_auc = roc_auc_score(y_test, y_pred)
                except Exception as exception:
                    roc_auc = None
                    if self.ignore_warnings == False:
                        print("ROC AUC couldn't be calculated for "+name)
                        print(exception)
                names.append(name)
                Accuracy.append(accuracy)
                B_Accuracy.append(b_accuracy)
                ROC_AUC.append(roc_auc)
                F1.append(f1)
                TIME.append(time.time() - start)
                if self.custom_metric != None:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)
                if self.verbose > 0:
                    if self.custom_metric != None:
                        print({"Model": name,
                               "Accuracy": accuracy,
                               "Balanced Accuracy": b_accuracy,
                               "ROC AUC": roc_auc,
                               "F1 Score": f1,
                               self.custom_metric.__name__: custom_metric,
                              "Time taken": time.time() - start})
                    else:
                        print({"Model": name,
                               "Accuracy": accuracy,
                               "Balanced Accuracy": b_accuracy,
                               "ROC AUC": roc_auc,
                               "F1 Score": f1,
                              "Time taken": time.time() - start})
                if self.predictions == True:
                    predictions[name]=y_pred
            except Exception as exception:
                if self.ignore_warnings == False:
                    print(name + " model failed to execute")
                    print(exception)
        if self.custom_metric == None:
            scores = pd.DataFrame({"Model": names,
                                   "Accuracy": Accuracy,
                                   "Balanced Accuracy": B_Accuracy,
                                   "ROC AUC": ROC_AUC,
                                   "F1 Score": F1,
                                   "Time Taken": TIME})
        else:
            scores = pd.DataFrame({"Model": names,
                                   "Accuracy": Accuracy,
                                   "Balanced Accuracy": B_Accuracy,
                                   "ROC AUC": ROC_AUC,
                                   "F1 Score": F1,
                                  self.custom_metric.__name__: CUSTOM_METRIC,
                                  "Time Taken": TIME})
        scores = scores.sort_values(
            by='Balanced Accuracy', ascending=False).set_index('Model')

        if self.predictions == True:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions == True else scores

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
    >>> reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
    >>> models,predictions = reg.fit(X_train, X_test, y_train, y_test)
    | Model                         |   R-Squared |     RMSE |   Time Taken |
    |:------------------------------|------------:|---------:|-------------:|
    | SVR                           |   0.877199  |  2.62054 |    0.0330021 |
    | RandomForestRegressor         |   0.874429  |  2.64993 |    0.0659981 |
    | ExtraTreesRegressor           |   0.867566  |  2.72138 |    0.0570002 |
    | AdaBoostRegressor             |   0.865851  |  2.73895 |    0.144999  |
    | NuSVR                         |   0.863712  |  2.7607  |    0.0340044 |
    | GradientBoostingRegressor     |   0.858693  |  2.81107 |    0.13      |
    | KNeighborsRegressor           |   0.826307  |  3.1166  |    0.0179954 |
    | HistGradientBoostingRegressor |   0.810479  |  3.25551 |    0.820995  |
    | BaggingRegressor              |   0.800056  |  3.34383 |    0.0579946 |
    | MLPRegressor                  |   0.750536  |  3.73503 |    0.725997  |
    | HuberRegressor                |   0.736973  |  3.83522 |    0.0370018 |
    | LinearSVR                     |   0.71914   |  3.9631  |    0.0179989 |
    | RidgeCV                       |   0.718402  |  3.9683  |    0.018003  |
    | BayesianRidge                 |   0.718102  |  3.97041 |    0.0159984 |
    | Ridge                         |   0.71765   |  3.9736  |    0.0149941 |
    | LinearRegression              |   0.71753   |  3.97444 |    0.0190051 |
    | TransformedTargetRegressor    |   0.71753   |  3.97444 |    0.012001  |
    | LassoCV                       |   0.717337  |  3.9758  |    0.0960066 |
    | ElasticNetCV                  |   0.717104  |  3.97744 |    0.0860076 |
    | LassoLarsCV                   |   0.717045  |  3.97786 |    0.0490005 |
    | LassoLarsIC                   |   0.716636  |  3.98073 |    0.0210001 |
    | LarsCV                        |   0.715031  |  3.99199 |    0.0450008 |
    | Lars                          |   0.715031  |  3.99199 |    0.0269964 |
    | SGDRegressor                  |   0.714362  |  3.99667 |    0.0210009 |
    | RANSACRegressor               |   0.707849  |  4.04198 |    0.111998  |
    | ElasticNet                    |   0.690408  |  4.16088 |    0.0190012 |
    | Lasso                         |   0.662141  |  4.34668 |    0.0180018 |
    | OrthogonalMatchingPursuitCV   |   0.591632  |  4.77877 |    0.0180008 |
    | ExtraTreeRegressor            |   0.583314  |  4.82719 |    0.0129974 |
    | PassiveAggressiveRegressor    |   0.556668  |  4.97914 |    0.0150032 |
    | GaussianProcessRegressor      |   0.428298  |  5.65425 |    0.0580051 |
    | OrthogonalMatchingPursuit     |   0.379295  |  5.89159 |    0.0180039 |
    | DecisionTreeRegressor         |   0.318767  |  6.17217 |    0.0230272 |
    | DummyRegressor                |  -0.0215752 |  7.55832 |    0.0140116 |
    | LassoLars                     |  -0.0215752 |  7.55832 |    0.0180008 |
    | KernelRidge                   |  -8.24669   | 22.7396  |    0.0309792 |
    """

    def __init__(self, verbose=0, ignore_warnings=True, custom_metric = None, predictions = False):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions

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
        RMSE = []
        # WIN = []
        names = []
        TIME = []
        predictions = {}
        
        if self.custom_metric != None:
            CUSTOM_METRIC = []

        if type(X_train) is np.ndarray:
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(
            include=['int64', 'float64', 'int32', 'float32']).columns
        categorical_features = X_train.select_dtypes(
            include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features)
            ])

        for name, model in tqdm(REGRESSORS):
            start = time.time()
            try:
                pipe = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', model())
                ])
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                r_squared = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                names.append(name)
                R2.append(r_squared)
                RMSE.append(rmse)
                TIME.append(time.time() - start)
                if self.custom_metric != None:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)

                if self.verbose > 0:
                    if self.custom_metric != None:
                        print({"Model": name,
                               "R-Squared": r_squared,
                               "RMSE": rmse,
                               self.custom_metric.__name__: custom_metric,
                              "Time taken": time.time() - start})
                    else:
                        print({"Model": name,
                               "R-Squared": r_squared,
                               "RMSE": rmse,
                              "Time taken": time.time() - start})
                if self.predictions == True:
                    predictions[name]=y_pred
            except Exception as exception:
                if self.ignore_warnings == False:
                    print(name + " model failed to execute")
                    print(exception)
                    
        if self.custom_metric == None:
            scores = pd.DataFrame({"Model": names, 
                                   "R-Squared": R2, 
                                   "RMSE": RMSE,
                                   "Time Taken": TIME})
        else:
            scores = pd.DataFrame({"Model": names, 
                                   "R-Squared": R2, 
                                   "RMSE": RMSE,
                                  self.custom_metric.__name__: CUSTOM_METRIC,
                                  "Time Taken": TIME})
        scores = scores.sort_values(
            by='R-Squared', ascending=False).set_index('Model')
        
        if self.predictions == True:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions == True else scores

Regression = LazyRegressor
Classification = LazyClassifier