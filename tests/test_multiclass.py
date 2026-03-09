import pytest
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def test_multiclass_classification():
    """Test that LazyClassifier works with multiclass datasets."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, preds = clf.fit(X_train, X_test, y_train, y_test)
    assert len(models) > 0
    assert "Balanced Accuracy" in models.columns
