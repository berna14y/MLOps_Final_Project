# preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import os

def preprocess_data(df):
    """
    Perform all preprocessing steps on the raw data.
    """
    # Feature engineering
    df_fe = df.copy()
    df_fe.drop(columns=['duration'], inplace=True)
    df_fe['contacted_before'] = df_fe['pdays'].apply(lambda x: 0 if x == 999 else 1)

    # Age binning (if not already done in EDA)
    if 'age_group' not in df_fe.columns:
        df_fe['age_group'] = pd.cut(df_fe['age'], bins=[0, 25, 35, 45, 55, 65, 100],
                                labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'], right=False)
    df_fe.drop(columns=['age'], inplace=True)

    # Categorical columns to label encode
    categorical_cols = [
        'job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'day_of_week', 'poutcome', 'age_group'
    ]

    # Apply Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_fe[col] = le.fit_transform(df_fe[col])
        label_encoders[col] = le

    # Encode target
    df_fe['y'] = df_fe['y'].map({'no': 0, 'yes': 1})

    return df_fe, label_encoders

def handle_class_imbalance(X_train, y_train):
    """
    Handle class imbalance using combined under and over sampling.
    """
    under = RandomUnderSampler(sampling_strategy=0.2, random_state=42)
    over = SMOTE(sampling_strategy=0.5, random_state=42)
    resample_pipeline = Pipeline([('under', under), ('over', over)])
    X_train_resampled, y_train_resampled = resample_pipeline.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled

def check_multicollinearity(X_train_df):
    """
    Check for multicollinearity using correlation and VIF.
    Returns features to drop.
    """
    # Correlation matrix
    correlation_matrix = X_train_df.corr()

    # Identify high correlation pairs
    high_corr_pairs = []
    cols = correlation_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) >= 0.7:
                high_corr_pairs.append((cols[i], cols[j], corr))

    # Prepare VIF DataFrame
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_train_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train_df.values, i) for i in range(X_train_df.shape[1])]

    return high_corr_pairs, vif_data

def get_features_to_drop(df_fe, high_corr_pairs):
    """
    Determine which features to drop based on multicollinearity and target relevance.
    """
    correlations_with_target = df_fe.corr()['y'].drop('y').abs()
    correlated_pairs = [(pair[0], pair[1]) for pair in high_corr_pairs]

    features_to_drop = []
    for feat1, feat2 in correlated_pairs:
        corr1 = correlations_with_target.get(feat1, 0)
        corr2 = correlations_with_target.get(feat2, 0)
        drop_feat = feat1 if corr1 < corr2 else feat2
        features_to_drop.append(drop_feat)

    return list(set(features_to_drop))

def prepare_data(df_path='bank-additional-full.csv'):
    """
    Main function to prepare data for modeling.
    Returns preprocessed and split data.
    """
    # Load and preprocess
    df = pd.read_csv(df_path, sep=';')
    df_fe, label_encoders = preprocess_data(df)

    # Split data
    X = df_fe.drop(columns=['y'])
    y = df_fe['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle class imbalance
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)

    # Check multicollinearity
    X_train_df = pd.DataFrame(X_train_resampled, columns=X.columns)
    high_corr_pairs, vif_data = check_multicollinearity(X_train_df)
    features_to_drop = get_features_to_drop(df_fe, high_corr_pairs)

    # Final feature selection
    X_train_final = X_train_df.drop(columns=features_to_drop)
    X_test_final = pd.DataFrame(X_test, columns=X.columns).drop(columns=features_to_drop)

    # Retrain scaler only on final model features
    scaler = RobustScaler()
    scaler.fit(X_train_final)

    # Scale
    X_train_final_scaled = pd.DataFrame(scaler.transform(X_train_final), columns=X_train_final.columns)
    X_test_final_scaled = pd.DataFrame(scaler.transform(X_test_final), columns=X_test_final.columns)

    os.makedirs('outputs', exist_ok=True)

    joblib.dump(label_encoders, 'outputs/label_encoders.pkl')
    joblib.dump(scaler, 'outputs/scaler.pkl')

    return {
        'X_train': X_train_final_scaled,
        'X_test': X_test_final_scaled,
        'y_train': y_train_resampled,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'features_to_drop': features_to_drop,
        'high_corr_pairs': high_corr_pairs,
        'vif_data': vif_data
    }

if __name__ == "__main__":
    import sys
    from contextlib import redirect_stdout

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/outputs_preprocess.txt", "w") as f:
        with redirect_stdout(f):
            data_dict = prepare_data()

            print("📌 Features to Drop Due to Multicollinearity:\n")
            for feat in data_dict['features_to_drop']:
                print("-", feat)

            print("\n🔗 High Correlation Feature Pairs (|r| ≥ 0.7):\n")
            for f1, f2, corr in data_dict['high_corr_pairs']:
                print(f"{f1} ↔ {f2} | Correlation: {corr:.3f}")

            print("\n📈 VIF Table:\n")
            print(data_dict['vif_data'].to_string(index=False))
