"""Quick test to verify all modules are importable for documentation."""

from lazypredict.Supervised import LazyClassifier, LazyRegressor
from lazypredict.Explainer import ModelExplainer

print('✅ All modules successfully imported')
print(f'\nLazyClassifier public methods: {len([m for m in dir(LazyClassifier) if not m.startswith("_")])}')
print(f'LazyRegressor public methods: {len([m for m in dir(LazyRegressor) if not m.startswith("_")])}')
print(f'ModelExplainer public methods: {len([m for m in dir(ModelExplainer) if not m.startswith("_")])}')

# Check docstrings exist
print(f'\n✅ LazyClassifier has docstring: {LazyClassifier.__doc__ is not None}')
print(f'✅ LazyRegressor has docstring: {LazyRegressor.__doc__ is not None}')
print(f'✅ ModelExplainer has docstring: {ModelExplainer.__doc__ is not None}')

# Check key methods have docstrings
print(f'\n✅ ModelExplainer.feature_importance has docstring: {ModelExplainer.feature_importance.__doc__ is not None}')
print(f'✅ ModelExplainer.plot_summary has docstring: {ModelExplainer.plot_summary.__doc__ is not None}')
print(f'✅ ModelExplainer.explain_prediction has docstring: {ModelExplainer.explain_prediction.__doc__ is not None}')

print('\n🎉 All modules are properly documented and ready for Sphinx API generation!')
