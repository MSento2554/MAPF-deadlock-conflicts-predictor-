import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Directory configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    os.path.dirname(current_dir),
    "congestion_model",
    "processed_congestion_dataset.csv",
)
MODEL_DIR = os.path.join(current_dir, "models")
FIGURES_DIR = os.path.join(current_dir, "figures")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def train_temporal_model():
    # Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # Prepare target labels
    df = df.dropna(subset=["target_future_conflict"])
    df["target_future_conflict"] = (
        df["target_future_conflict"].astype(int)
    )

    # Sort by simulation time
    df = df.sort_values(
        ["t", "robot_id"]
    ).reset_index(drop=True)

    # Temporal feature set
    temporal_feature_cols = [
        "t",
        "steps_stationary",
        "stationary_ratio",
        "neighbour_static_count",
        "density_3_delta",
        "density_5_delta",
        "density_3_accel",
        "density_5_accel",
        "density_3_lag_1",
        "density_5_lag_1",
        "density_3_lag_2",
        "density_5_lag_2",
        "density_3_lag_3",
        "density_5_lag_3",
    ]

    available_features = [
        col
        for col in temporal_feature_cols
        if col in df.columns
    ]

    # Time-based train/test split
    split_tick = df["t"].quantile(0.8)

    train_df = df[df["t"] <= split_tick]
    test_df = df[df["t"] > split_tick]

    X_train_full = train_df[available_features]
    y_train_full = train_df["target_future_conflict"]

    X_test = test_df[available_features]
    y_test = test_df["target_future_conflict"]

    # 5-fold TimeSeries Cross Validation on training portion
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = []
    thresholds = []

    print("Running TimeSeries Cross Validation...")
    for fold, (train_idx, val_idx) in enumerate(
        tscv.split(X_train_full),
        start=1,
    ):

        X_tr = X_train_full.iloc[train_idx]
        y_tr = y_train_full.iloc[train_idx]

        X_val = X_train_full.iloc[val_idx]
        y_val = y_train_full.iloc[val_idx]

        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()

        scale_pos_weight = neg / pos if pos > 0 else 1.0

        cv_model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        )

        cv_model.fit(X_tr, y_tr)

        prob = cv_model.predict_proba(X_val)[:, 1]

        roc = roc_auc_score(y_val, prob)
        pr = average_precision_score(y_val, prob)

        best_thr = 0.5
        best_f1 = 0

        for thr in np.arange(0.01, 1.00, 0.01):

            pred = (prob >= thr).astype(int)

            score = f1_score(
                y_val,
                pred,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_thr = thr

        pred = (prob >= best_thr).astype(int)

        acc = accuracy_score(y_val, pred)
        prec = precision_score(
            y_val,
            pred,
            zero_division=0,
        )
        rec = recall_score(
            y_val,
            pred,
            zero_division=0,
        )

        cv_results.append(
            {
                "acc": acc,
                "precision": prec,
                "recall": rec,
                "f1": best_f1,
                "roc": roc,
                "pr": pr,
            }
        )

        thresholds.append(best_thr)

        print(
            f"Fold {fold}: "
            f"Acc={acc:.4f} "
            f"Prec={prec:.4f} "
            f"Recall={rec:.4f} "
            f"F1={best_f1:.4f} "
            f"ROC={roc:.4f}"
        )

    # Cross Validation Summary
    print("\n===== Cross Validation Summary =====")
    for key in [
        "acc",
        "precision",
        "recall",
        "f1",
        "roc",
        "pr",
    ]:

        scores = [x[key] for x in cv_results]

        print(
            f"{key.upper():10s}: "
            f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"
        )
    
    optimal_threshold = np.mean(thresholds)
    print(
        f"Threshold : {optimal_threshold:.3f}"
    )

    # Handle class imbalance for final model
    neg = (y_train_full == 0).sum()
    pos = (y_train_full == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # Configure final XGBoost model
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )

    print(
        "\nTraining final Temporal XGBoost model on full train set..."
    )

    model.fit(
        X_train_full,
        y_train_full,
    )

    # Evaluate on test set
    prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, prob)
    pr = average_precision_score(y_test, prob)

    final_pred = (
        prob >= optimal_threshold
    ).astype(int)

    print("\n=== Evaluation Metrics on Test Set ===")
    print(
        classification_report(
            y_test,
            final_pred,
            zero_division=0,
        )
    )
    print(f"ROC-AUC           : {roc:.4f}")
    print(f"PR-AUC            : {pr:.4f}")
    print(f"Applied Threshold : {optimal_threshold:.3f}")
    print(f"Final F1-Score    : {f1_score(y_test, final_pred, zero_division=0):.4f}")

    # Save model package
    model_path = os.path.join(
        MODEL_DIR,
        "temporal_model.pkl",
    )

    joblib.dump(
        {
            "model": model,
            "threshold": optimal_threshold,
            "features": available_features,
        },
        model_path,
    )

    print(f"\nModel saved to: {model_path}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))

    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.barh(
        range(len(indices)),
        importances[indices],
        color="darkorange",
        align="center",
    )

    plt.yticks(
        range(len(indices)),
        [available_features[i] for i in indices],
    )

    plt.xlabel("Feature Importance")
    plt.title("Temporal Model Feature Importance")

    figure_path = os.path.join(
        FIGURES_DIR,
        "temporal_feature_importance.png",
    )

    plt.savefig(
        figure_path,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Feature importance plot saved to: {figure_path}"
    )


if __name__ == "__main__":
    train_temporal_model()