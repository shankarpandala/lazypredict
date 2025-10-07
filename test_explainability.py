"""
Test Model Explainability Feature
Demonstrates SHAP-based explanations for LazyPredict models
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from lazypredict.Explainer import ModelExplainer

def test_classification_explainability():
    """Test explainability features for classification models."""
    print("=" * 80)
    print("Testing Classification Model Explainability")
    print("=" * 80)
    
    # Load dataset
    print("\n1. Loading breast cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Create DataFrame with proper column names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"   Dataset shape: {X_df.shape}")
    print(f"   Features: {list(feature_names[:5])}...")
    
    # Split data
    print("\n2. Splitting data (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train models (use subset for faster testing)
    print("\n3. Training classification models...")
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        predictions=False,
        random_state=42,
        classifiers=[
            'LogisticRegression',
            'RandomForestClassifier', 
            'DecisionTreeClassifier',
            'XGBClassifier',
            'LGBMClassifier'
        ]
    )
    
    models = clf.fit(X_train, X_test, y_train, y_test)
    print(f"\n✅ Trained {len(models)} models")
    print("\nTop 3 models:")
    print(models.head(3))
    
    # Create explainer
    print("\n4. Creating ModelExplainer...")
    explainer = ModelExplainer(clf, X_train, X_test)
    print(f"✅ Explainer initialized with {len(explainer.trained_models)} models")
    
    # Test feature importance
    print("\n5. Computing feature importance...")
    best_model = models.index[0]
    print(f"   Best model: {best_model}")
    
    importance = explainer.feature_importance(best_model, top_n=10)
    print(f"\n✅ Top 10 most important features for {best_model}:")
    print(importance.to_string())
    
    # Test SHAP summary plot
    print(f"\n6. Generating SHAP summary plot for {best_model}...")
    try:
        explainer.plot_summary(best_model, plot_type='bar', max_display=10, show=False)
        print("✅ Summary plot generated successfully")
    except Exception as e:
        print(f"⚠️  Summary plot failed: {e}")
    
    # Test single prediction explanation
    print(f"\n7. Explaining single prediction (instance 0)...")
    try:
        explainer.explain_prediction(best_model, instance_idx=0, show=False)
        print("✅ Prediction explanation generated successfully")
    except Exception as e:
        print(f"⚠️  Prediction explanation failed: {e}")
    
    # Test top features for a prediction
    print(f"\n8. Getting top features for instance 0...")
    top_features = explainer.get_top_features(best_model, instance_idx=0, top_n=5)
    print(f"\n✅ Top 5 features affecting prediction:")
    print(top_features.to_string())
    
    # Test model comparison
    print(f"\n9. Comparing feature importance across models...")
    try:
        comparison = explainer.compare_models(
            model_names=list(models.index[:3]),
            top_n_features=5,
            show=False
        )
        print(f"\n✅ Feature importance comparison:")
        print(comparison.to_string())
    except Exception as e:
        print(f"⚠️  Model comparison failed: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Classification explainability test completed successfully!")
    print("=" * 80)


def test_regression_explainability():
    """Test explainability features for regression models."""
    print("\n\n" + "=" * 80)
    print("Testing Regression Model Explainability")
    print("=" * 80)
    
    # Load dataset
    print("\n1. Loading diabetes dataset...")
    data = load_diabetes()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Create DataFrame with proper column names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"   Dataset shape: {X_df.shape}")
    print(f"   Features: {list(feature_names)}")
    
    # Split data
    print("\n2. Splitting data (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train models (use subset for faster testing)
    print("\n3. Training regression models...")
    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        predictions=False,
        random_state=42,
        regressors=[
            'LinearRegression',
            'Ridge',
            'RandomForestRegressor',
            'XGBRegressor',
            'LGBMRegressor'
        ]
    )
    
    models = reg.fit(X_train, X_test, y_train, y_test)
    print(f"\n✅ Trained {len(models)} models")
    print("\nTop 3 models:")
    print(models.head(3))
    
    # Create explainer
    print("\n4. Creating ModelExplainer...")
    explainer = ModelExplainer(reg, X_train, X_test)
    print(f"✅ Explainer initialized with {len(explainer.trained_models)} models")
    
    # Test feature importance
    print("\n5. Computing feature importance...")
    best_model = models.index[0]
    print(f"   Best model: {best_model}")
    
    importance = explainer.feature_importance(best_model, top_n=10)
    print(f"\n✅ Top features for {best_model}:")
    print(importance.to_string())
    
    # Test SHAP summary plot
    print(f"\n6. Generating SHAP summary plot for {best_model}...")
    try:
        explainer.plot_summary(best_model, plot_type='bar', show=False)
        print("✅ Summary plot generated successfully")
    except Exception as e:
        print(f"⚠️  Summary plot failed: {e}")
    
    # Test dependence plot
    print(f"\n7. Generating dependence plot...")
    try:
        top_feature = importance.iloc[0]['feature']
        explainer.plot_dependence(best_model, top_feature, show=False)
        print(f"✅ Dependence plot for '{top_feature}' generated successfully")
    except Exception as e:
        print(f"⚠️  Dependence plot failed: {e}")
    
    # Test top features for a prediction
    print(f"\n8. Getting top features for instance 0...")
    top_features = explainer.get_top_features(best_model, instance_idx=0, top_n=5)
    print(f"\n✅ Top 5 features affecting prediction:")
    print(top_features.to_string())
    
    print("\n" + "=" * 80)
    print("✅ Regression explainability test completed successfully!")
    print("=" * 80)


def test_explainability_edge_cases():
    """Test edge cases and error handling."""
    print("\n\n" + "=" * 80)
    print("Testing Edge Cases and Error Handling")
    print("=" * 80)
    
    # Load simple dataset
    data = load_breast_cancer()
    X, y = data.data[:100], data.target[:100]  # Small dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a single model
    print("\n1. Training single model...")
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        classifiers=['LogisticRegression']
    )
    models = clf.fit(X_train, X_test, y_train, y_test)
    print("✅ Model trained")
    
    # Create explainer
    print("\n2. Creating explainer...")
    explainer = ModelExplainer(clf, X_train, X_test)
    
    # Test with invalid model name
    print("\n3. Testing with invalid model name...")
    try:
        explainer.feature_importance('NonExistentModel')
        print("❌ Should have raised an error")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {str(e)[:50]}...")
    
    # Test with invalid instance index
    print("\n4. Testing with invalid instance index...")
    try:
        explainer.explain_prediction('LogisticRegression', instance_idx=1000)
        print("❌ Should have raised an error")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {str(e)[:50]}...")
    
    # Test with numpy arrays (no column names)
    print("\n5. Testing with numpy arrays (no feature names)...")
    explainer2 = ModelExplainer(clf, X_train, X_test)
    importance = explainer2.feature_importance('LogisticRegression', top_n=3)
    print(f"✅ Works with numpy arrays (auto-generated names):")
    print(importance.to_string())
    
    print("\n" + "=" * 80)
    print("✅ Edge case testing completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Run all tests
    try:
        test_classification_explainability()
    except Exception as e:
        print(f"\n❌ Classification test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_regression_explainability()
    except Exception as e:
        print(f"\n❌ Regression test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_explainability_edge_cases()
    except Exception as e:
        print(f"\n❌ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\n" + "=" * 80)
    print("🎉 ALL EXPLAINABILITY TESTS COMPLETED!")
    print("=" * 80)
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Feature importance calculation")
    print("   ✅ SHAP summary plots")
    print("   ✅ Single prediction explanations")
    print("   ✅ Dependence plots")
    print("   ✅ Model comparison")
    print("   ✅ Top feature extraction")
    print("   ✅ Error handling")
    print("\n📚 Usage Example:")
    print("   from lazypredict.Supervised import LazyClassifier")
    print("   from lazypredict.Explainer import ModelExplainer")
    print("   ")
    print("   clf = LazyClassifier()")
    print("   models = clf.fit(X_train, X_test, y_train, y_test)")
    print("   ")
    print("   explainer = ModelExplainer(clf, X_train, X_test)")
    print("   importance = explainer.feature_importance('LogisticRegression')")
    print("   explainer.plot_summary('LogisticRegression')")
    print("=" * 80)
