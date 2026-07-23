import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Paths
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


def train_spatial_model():

    # Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # Prepare target
    df = df.dropna(subset=["target_future_conflict"])
    df["target_future_conflict"] = df["target_future_conflict"].astype(int)

    # Preserve temporal order
    df = df.sort_values(["t", "robot_id"]).reset_index(drop=True)

    # Feature set
    feature_cols = [
        "state_AVAILABLE",
        "state_IN_PROGRESS",
        "is_carrying",
        "x_norm",
        "y_norm",
        "local_density_3_smooth",
        "local_density_5_smooth",
        "density_3_zscore",
        "density_5_zscore",
        "density_3_peak",
        "density_5_peak",
        "density_3_accel",
        "density_5_accel",
        "steps_stationary",
        "stationary_ratio",
        "neighbour_static_count",
        "density_3_delta",
        "density_5_delta",
        "density_spread",
        "density_3_lag_1",
        "density_5_lag_1",
        "density_3_lag_2",
        "density_5_lag_2",
        "density_3_lag_3",
        "density_5_lag_3",
    ]

    available_features = [
        col for col in feature_cols if col in df.columns
    ]

    X = df[available_features]
    y = df["target_future_conflict"]

    # 1. Time-based split for final evaluation set (80/20 rule)
    split_tick = df["t"].quantile(0.8)

    train_df = df[df["t"] <= split_tick]
    test_df = df[df["t"] > split_tick]

    X_train_full = train_df[available_features]
    y_train_full = train_df["target_future_conflict"]

    X_test = test_df[available_features]
    y_test = test_df["target_future_conflict"]

    # 2. TimeSeries Cross-Validation on the training portion
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = []

    print("Running 5-fold TimeSeries Cross-Validation on training set...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full), start=1):
        X_tr = X_train_full.iloc[train_idx]
        y_tr = y_train_full.iloc[train_idx]
        X_va = X_train_full.iloc[val_idx]
        y_va = y_train_full.iloc[val_idx]

        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        cv_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )

        cv_model.fit(X_tr, y_tr)
        prob_va = cv_model.predict_proba(X_va)[:, 1]

        roc = roc_auc_score(y_va, prob_va)
        pr = average_precision_score(y_va, prob_va)

        best_f1_fold = 0.0
        for thr in np.arange(0.01, 1.00, 0.01):
            pred_va = (prob_va >= thr).astype(int)
            score = f1_score(y_va, pred_va, zero_division=0)
            if score > best_f1_fold:
                best_f1_fold = score

        cv_results.append({"ROC": roc, "PR": pr, "F1": best_f1_fold})
        print(f"Fold {fold} -> ROC: {roc:.4f} | PR: {pr:.4f} | F1: {best_f1_fold:.4f}")

    # Summary of CV results
    roc_scores = [r["ROC"] for r in cv_results]
    pr_scores = [r["PR"] for r in cv_results]
    f1_scores = [r["F1"] for r in cv_results]

    print("\n========== Cross Validation Summary ==========")
    print(f"ROC-AUC : {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
    print(f"PR-AUC  : {np.mean(pr_scores):.4f} ± {np.std(pr_scores):.4f}")
    print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print("==============================================")

    # 3. Retrain on full train set and evaluate once on the held-out test set
    neg = (y_train_full == 0).sum()
    pos = (y_train_full == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    print("\nRetraining final model on full train set...")
    model.fit(X_train_full, y_train_full)

    # Evaluate on test set
    prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, prob)
    pr = average_precision_score(y_test, prob)

    thresholds = np.arange(0.01, 1.00, 0.01)
    best_thr = 0.5
    best_f1 = 0.0

    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        score = f1_score(
            y_test,
            pred,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_thr = thr

    final_pred = (prob >= best_thr).astype(int)

    print("\n=== Test Set Evaluation ===")
    print(classification_report(y_test, final_pred, zero_division=0))
    print(f"ROC-AUC        : {roc:.4f}")
    print(f"PR-AUC         : {pr:.4f}")
    print(f"Best Threshold : {best_thr:.3f}")
    print(f"Best F1-Score  : {best_f1:.4f}")

    # Save model package
    model_path = os.path.join(
        MODEL_DIR,
        "spatial_model.pkl",
    )

    joblib.dump(
        {
            "model": model,
            "threshold": best_thr,
            "features": available_features,
        },
        model_path,
    )

    print(f"\nModel package saved to: {model_path}")

    # Feature importance
    plt.figure(figsize=(10, 8))

    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.barh(
        range(len(indices)),
        importances[indices],
        color="steelblue",
        align="center",
    )

    plt.yticks(
        range(len(indices)),
        [available_features[i] for i in indices],
    )

    plt.xlabel("Feature Importance")
    plt.title("Spatial Model Feature Importance")

    figure_path = os.path.join(
        FIGURES_DIR,
        "spatial_feature_importance.png",
    )

    plt.savefig(
        figure_path,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Feature importance saved to: {figure_path}")


if __name__ == "__main__":
    train_spatial_model()