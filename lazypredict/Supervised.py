"""
Supervised Models
"""
# Author: Shankar Rao Pandala <shankar.pandala@livee.com>

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.testing import all_estimators
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings("ignore")


CLASSIFIERS = [est for est in all_estimators(
) if issubclass(est[1], ClassifierMixin)]
REGRESSORS = [est for est in all_estimators(
) if issubclass(est[1], RegressorMixin)]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoding', OneHotEncoder(handle_unknown='ignore'))
])

# Helper class for performing classification


class Classification:
    """In machine learning and statistics, classification is the problem of identifying 
    to which of a set of categories (sub-populations) a new observation belongs, on the 
    basis of a training set of data containing observations (or instances) whose category 
    membership is known. Examples are assigning a given email to the "spam" or "non-spam" 
    class, and assigning a diagnosis to a given patient based on observed characteristics 
    of the patient (sex, blood pressure, presence or absence of certain symptoms, etc.). 
    Classification is an example of pattern recognition.

    In the terminology of machine learning, classification is considered an instance of 
    supervised learning, i.e., learning where a training set of correctly identified 
    observations is available. The corresponding unsupervised procedure is known as 
    clustering, and involves grouping data into categories based on some measure of 
    inherent similarity or distance.
    Parameters
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(random_state=0, solver='lbfgs',
    ...                          multi_class='multinomial').fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :]) # doctest: +ELLIPSIS
    array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
           [9.7...e-01, 2.8...e-02, ...e-08]])
    >>> clf.score(X, y)
    0.97...
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_breast_cancer()
    >>> X = data.data
    >>> y= data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
    >>> clf = Classification()
    >>> clf.fit(X_train, X_test, y_train, y_test)
    | Model                          |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |
    |:-------------------------------|-----------:|--------------------:|----------:|-----------:|
    | LinearSVC                      |   0.989474 |            0.987544 |  0.987544 |   0.989462 |
    | Perceptron                     |   0.985965 |            0.984797 |  0.984797 |   0.985965 |
    | MLPClassifier                  |   0.985965 |            0.984797 |  0.984797 |   0.985965 |
    | LogisticRegressionCV           |   0.985965 |            0.98269  |  0.98269  |   0.985934 |
    | LogisticRegression             |   0.985965 |            0.98269  |  0.98269  |   0.985934 |
    | SVC                            |   0.982456 |            0.979942 |  0.979942 |   0.982437 |
    | PassiveAggressiveClassifier    |   0.982456 |            0.977835 |  0.977835 |   0.982398 |
    | CalibratedClassifierCV         |   0.982456 |            0.975728 |  0.975728 |   0.982357 |
    | GradientBoostingClassifier     |   0.978947 |            0.975088 |  0.975088 |   0.978902 |
    | LabelPropagation               |   0.975439 |            0.974448 |  0.974448 |   0.975464 |
    | LabelSpreading                 |   0.975439 |            0.974448 |  0.974448 |   0.975464 |
    | RandomForestClassifier         |   0.97193  |            0.971701 |  0.971701 |   0.971987 |
    | SGDClassifier                  |   0.97193  |            0.967486 |  0.967486 |   0.971869 |
    | QuadraticDiscriminantAnalysis  |   0.964912 |            0.966206 |  0.966206 |   0.965052 |
    | HistGradientBoostingClassifier |   0.968421 |            0.964739 |  0.964739 |   0.968387 |
    | RidgeClassifierCV              |   0.97193  |            0.963272 |  0.963272 |   0.971736 |
    | GaussianProcessClassifier      |   0.968421 |            0.960525 |  0.960525 |   0.968242 |
    | RidgeClassifier                |   0.968421 |            0.960525 |  0.960525 |   0.968242 |
    | AdaBoostClassifier             |   0.961404 |            0.959245 |  0.959245 |   0.961444 |
    | ExtraTreesClassifier           |   0.961404 |            0.957138 |  0.957138 |   0.961362 |
    | KNeighborsClassifier           |   0.961404 |            0.95503  |  0.95503  |   0.961276 |
    | BaggingClassifier              |   0.950877 |            0.95311  |  0.95311  |   0.951161 |
    | BernoulliNB                    |   0.950877 |            0.951003 |  0.951003 |   0.951072 |
    | LinearDiscriminantAnalysis     |   0.961404 |            0.950816 |  0.950816 |   0.961089 |
    | GaussianNB                     |   0.954386 |            0.949536 |  0.949536 |   0.954337 |
    | NuSVC                          |   0.954386 |            0.943215 |  0.943215 |   0.954014 |
    | ExtraTreeClassifier            |   0.940351 |            0.934333 |  0.934333 |   0.940287 |
    | NearestCentroid                |   0.947368 |            0.933506 |  0.933506 |   0.946801 |
    | DecisionTreeClassifier         |   0.933333 |            0.928838 |  0.928838 |   0.933403 |
    | CheckingClassifier             |   0.361404 |            0.5      |  0.5      |   0.191879 |
    | DummyClassifier                |   0.487719 |            0.45351  |  0.45351  |   0.491539 |
    """

    def __init__(self, verbose=0, ignore_warnings=True):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings

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
        """
        Accuracy = []
        B_Accuracy = []
        ROC_AUC = []
        F1 = []
        names = []

        if type(X_train) is np.ndarray:
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(
            include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(
            include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features)
            ])

        for name, model in tqdm(CLASSIFIERS):
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
                if self.verbose > 0:
                    print({"Model": name,
                           "Accuracy": accuracy,
                           "Balanced Accuracy": b_accuracy,
                           "ROC AUC": roc_auc,
                           "F1 Score": f1})
            except Exception as exception:
                if self.ignore_warnings == False:
                    print(name + " model failed to execute")
                    print(exception)
        scores = pd.DataFrame({"Model": names,
                               "Accuracy": Accuracy,
                               "Balanced Accuracy": B_Accuracy,
                               "ROC AUC": ROC_AUC,
                               "F1 Score": F1})
        scores = scores.sort_values(
            by='Balanced Accuracy', ascending=False).set_index('Model')

        return scores
