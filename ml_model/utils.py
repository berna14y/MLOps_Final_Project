# utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_model(model, path):
    """
    Save trained model to file.
    """
    import joblib
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    """
    Load trained model from file.
    """
    import joblib
    return joblib.load(path)

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance from a tree-based model.
    """
    # For ensemble models, we'll use the first estimator (XGBoost in our case)
    if hasattr(model, 'named_estimators_'):
        model = model.named_estimators_['xgb']
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_score'):
        # For XGBoost
        importance_dict = model.get_booster().get_score(importance_type='weight')
        importances = np.array([importance_dict.get(f, 0) for f in feature_names])
    else:
        raise ValueError("Model doesn't have feature importance attribute")
    
    # Create DataFrame
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_imp)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return feature_imp