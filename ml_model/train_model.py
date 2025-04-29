# train_model.py
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

import joblib
import os
os.makedirs('outputs', exist_ok=True)

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train and evaluate the ensemble model.
    """
    # Create base models
    xgb = XGBClassifier(eval_metric='logloss',random_state=42,enable_categorical=False,use_label_encoder=False) # Explicitly set to False

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cat = CatBoostClassifier(verbose=0, random_state=42)

    # Voting Classifier
    voting_clf = VotingClassifier(estimators=[
        ('xgb', xgb),
        ('rf', rf),
        ('cat', cat)
    ], voting='soft')

    # Cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    roc_auc_scores = []

    for train_index, val_index in kf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Train Voting Classifier
        voting_clf.fit(X_tr, y_tr)

        # Predict
        y_pred = voting_clf.predict(X_val)
        y_proba = voting_clf.predict_proba(X_val)[:, 1]

        # Print results
        print(f"\n📊 Fold {fold} Classification Report:")
        print(classification_report(y_val, y_pred))

        auc = roc_auc_score(y_val, y_proba)
        roc_auc_scores.append(auc)
        print(f"ROC AUC Score: {auc:.4f}")

        fold += 1

    # Summary of all fold AUCs
    print("\n📈 ROC AUC Scores for Each Fold:")
    for i, auc in enumerate(roc_auc_scores, 1):
        print(f"Fold {i}: {auc:.4f}")



    # Summary
    print("\n🔁 Average ROC AUC Score Across Folds:", round(np.mean(roc_auc_scores), 4))
    
    # Final evaluation on test set
    print("\n🎯 Final Test Set Evaluation:")
    y_test_pred = voting_clf.predict(X_test)
    y_test_proba = voting_clf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_test_pred))
    print(f"Test ROC AUC Score: {roc_auc_score(y_test, y_test_proba):.4f}")

    # Save trained model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(voting_clf, 'outputs/voting_model.pkl')

    # Save features used in the model
    model_features = list(X_train.columns)
    joblib.dump(model_features, 'outputs/model_features.pkl')

    return voting_clf

def analyze_predictions(model, X_test, y_test):
    """
    Analyze model predictions and confidence.
    """
    # Get prediction probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Confidence histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(y_proba, bins=30, kde=True, color='skyblue')
    plt.title("Model Prediction Confidence Distribution")
    plt.xlabel("Predicted Probability (Confidence)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/prediction_confidence_histogram.png")
    plt.show()

    # Confidence vs correctness
    confidence_df = pd.DataFrame({
        "proba": y_proba,
        "pred": y_pred,
        "actual": y_test
    })
    confidence_df["correct"] = (confidence_df["pred"] == confidence_df["actual"]).astype(int)

    # Plot confidence of correct vs incorrect
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=confidence_df, x="correct", y="proba")
    plt.xticks([0, 1], ["Incorrect", "Correct"])
    plt.title("Confidence Comparison: Correct vs Incorrect Predictions")
    plt.ylabel("Predicted Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/confidence_correct_vs_incorrect.png")
    plt.show()

def explain_model(model, X_train, X_test, model_type='ensemble'):
    """
    Explain model predictions using SHAP values.
    """
    if model_type == 'xgb':
        explainer = shap.Explainer(model.named_estimators_['xgb'], X_train)
        tag = 'xgb'
    elif model_type == 'cat':
        explainer = shap.Explainer(model.named_estimators_['cat'], X_train)
        tag = 'cat'
    elif model_type == 'rf':
        explainer = shap.Explainer(model.named_estimators_['rf'], X_train)
        tag = 'rf'
    else:
        # Default to explaining XGBoost if no specific model selected
        explainer = shap.Explainer(model.named_estimators_['xgb'], X_train)
        tag = 'xgb'
    
    shap_values = explainer(X_test)
    
    # Global Feature Importance
    fig = shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(f"outputs/shap_beeswarm_{tag}.png")
    plt.close()
    
    # Local explanation for the first prediction
    fig = shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(f"outputs/shap_waterfall_{tag}.png")
    plt.close()
        # show=False prevents SHAP from displaying the plot immediately
        # plt.close() ensures the figure is closed after saving (prevents memory leaks)


def main():
    import sys
    from contextlib import redirect_stdout

    from preprocess import prepare_data

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/outputs_train.txt", "w") as f:
        with redirect_stdout(f):
            # Get preprocessed data
            print("📦 Loading and preprocessing data...\n")
            data_dict = prepare_data()

            # Train and evaluate model
            print("\n🤖 Training and cross-validating model...\n")
            model = train_and_evaluate(
                data_dict['X_train'], 
                data_dict['y_train'], 
                data_dict['X_test'], 
                data_dict['y_test']
            )

            # Analyze predictions
            print("\n📊 Analyzing prediction confidence...\n")
            analyze_predictions(model, data_dict['X_test'], data_dict['y_test'])

            # Explain model
            print("\n🔍 Explaining model predictions...\n")
            explain_model(model, data_dict['X_train'], data_dict['X_test'], model_type='xgb')
            explain_model(model, data_dict['X_train'], data_dict['X_test'], model_type='cat')
            explain_model(model, data_dict['X_train'], data_dict['X_test'], model_type='rf')


if __name__ == "__main__":
    main()