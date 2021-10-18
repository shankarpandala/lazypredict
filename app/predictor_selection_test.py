from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.model_selection import train_test_split
from sklearn import datasets


def main():
    # Load data
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    # Split in training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Lazy Classifier configuration with SPECIFIC classifier algorithms
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        classifiers=[
            RandomForestClassifier,
            LinearDiscriminantAnalysis,
            ExtraTreesClassifier,
            QuadraticDiscriminantAnalysis,
            SGDClassifier,
        ],
    )
    # Training and testing evaluation
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)


if __name__ == "__main__":
    main()
