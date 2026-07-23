import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

# Directory configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(current_dir), 'congestion_model', 'processed_congestion_dataset.csv')
MODEL_DIR = os.path.join(current_dir, 'models')
FIGURES_DIR = os.path.join(current_dir, 'figures')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def train_temporal_model():
    # Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # Generate target label
    df['is_congested'] = (df['local_density_5_smooth'] > 2.5).astype(int)

    # Temporal feature set
    temporal_feature_cols = [
        't',
        'steps_stationary',
        'stationary_ratio',
        'neighbour_static_count',
        'density_3_delta',
        'density_5_delta',
        'density_3_accel',
        'density_5_accel',
        'density_3_lag_1',
        'density_5_lag_1',
        'density_3_lag_2',
        'density_5_lag_2',
        'density_3_lag_3',
        'density_5_lag_3'
    ]

    available_features = [
        col for col in temporal_feature_cols
        if col in df.columns
    ]

    X = df[available_features]
    y = df['is_congested']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        eval_metric='logloss',
        random_state=42
    )

    print("Training Temporal XGBoost model...")
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)

    print("Temporal Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    # Save trained model
    model_path = os.path.join(MODEL_DIR, 'temporal_model.pkl')
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))

    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.barh(
        range(len(indices)),
        importances[indices],
        color='darkorange',
        align='center'
    )

    plt.yticks(
        range(len(indices)),
        [available_features[i] for i in indices]
    )

    plt.xlabel("Feature Importance")
    plt.title("Temporal Model Feature Importance")

    figure_path = os.path.join(
        FIGURES_DIR,
        'temporal_feature_importance.png'
    )

    plt.savefig(
        figure_path,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Feature importance plot saved to: {figure_path}")


if __name__ == "__main__":
    train_temporal_model()