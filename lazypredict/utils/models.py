# estimators/utils.py

from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin, RegressorMixin
import xgboost
import lightgbm

class ModelFetcher:
    """
    Utility class to fetch all compatible models from scikit-learn, 
    with options to include additional models from XGBoost and LightGBM.
    
    Methods
    -------
    fetch_classifiers(exclude=None)
        Returns a list of classifier models excluding specified models.
    fetch_regressors(exclude=None)
        Returns a list of regressor models excluding specified models.
    """

    @staticmethod
    def fetch_classifiers(exclude=None):
        """
        Fetches all available scikit-learn classifier models, excluding specified ones,
        and appends classifiers from XGBoost and LightGBM.

        Parameters
        ----------
        exclude : list of str, optional
            Names of classifiers to exclude from the list (default is None).

        Returns
        -------
        list of tuples
            List of tuples where each tuple contains the name of the classifier and the class itself.
        """
        if exclude is None:
            exclude = [
                "ClassifierChain", "ComplementNB", "GradientBoostingClassifier",
                "GaussianProcessClassifier", "HistGradientBoostingClassifier",
                "MLPClassifier", "LogisticRegressionCV", "MultiOutputClassifier",
                "MultinomialNB", "OneVsOneClassifier", "OneVsRestClassifier",
                "OutputCodeClassifier", "RadiusNeighborsClassifier", "VotingClassifier"
            ]
        
        classifiers = [
            (name, cls) for name, cls in all_estimators(type_filter='classifier')
            if issubclass(cls, ClassifierMixin) and name not in exclude
        ]
        
        # Append external libraries
        classifiers.append(("XGBClassifier", xgboost.XGBClassifier))
        classifiers.append(("LGBMClassifier", lightgbm.LGBMClassifier))
        return classifiers

    @staticmethod
    def fetch_regressors(exclude=None):
        """
        Fetches all available scikit-learn regressor models, excluding specified ones,
        and appends regressors from XGBoost and LightGBM.

        Parameters
        ----------
        exclude : list of str, optional
            Names of regressors to exclude from the list (default is None).

        Returns
        -------
        list of tuples
            List of tuples where each tuple contains the name of the regressor and the class itself.
        """
        if exclude is None:
            exclude = [
                "TheilSenRegressor", "ARDRegression", "CCA", "IsotonicRegression",
                "StackingRegressor", "MultiOutputRegressor", "MultiTaskElasticNet",
                "MultiTaskElasticNetCV", "MultiTaskLasso", "MultiTaskLassoCV",
                "PLSCanonical", "PLSRegression", "RadiusNeighborsRegressor",
                "RegressorChain", "VotingRegressor"
            ]
        
        regressors = [
            (name, cls) for name, cls in all_estimators(type_filter='regressor')
            if issubclass(cls, RegressorMixin) and name not in exclude
        ]
        
        # Append external libraries
        regressors.append(("XGBRegressor", xgboost.XGBRegressor))
        regressors.append(("LGBMRegressor", lightgbm.LGBMRegressor))
        return regressors
