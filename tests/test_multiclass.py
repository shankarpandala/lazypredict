from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, preds = clf.fit(X_train, X_test, y_train, y_test)
print('Multiclass works!')
print(f'Models tested: {len(models)}')
print(models.head())
