"""
Accuracy & Precision Evaluation for processed_congestion_dataset.csv
=====================================================================
Uses a time-based train/test split (80/20) to avoid temporal data leakage.
vertex_risk_factor is computed only from the training partition to prevent
target leakage.
"""

import os
import sys
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

# ── Config ─────────────────────────────────────────────────────────────────
CSV_PATH   = os.path.join(os.path.dirname(__file__), "processed_congestion_dataset.csv")
TARGET_COL = "target_future_conflict"
EXCLUDE_COLS = {"t", "robot_id", TARGET_COL}
# ──────────────────────────────────────────────────────────────────────────


def load_and_split(csv_path):
    print(f"\n[1/4] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"      Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"      Target balance  0: {(df[TARGET_COL]==0).sum():,}  "
          f"1: {(df[TARGET_COL]==1).sum():,}  "
          f"({df[TARGET_COL].mean()*100:.1f}% positive)")

    # Time-based 80/20 split — no shuffling to preserve temporal order
    split_tick = df["t"].max() * 0.8
    train_df = df[df["t"] <= split_tick].copy()
    test_df  = df[df["t"] >  split_tick].copy()

    print(f"\n      Train samples: {len(train_df):,}  (t <= {split_tick:.0f})")
    print(f"      Test  samples: {len(test_df):,}   (t >  {split_tick:.0f})")
    return train_df, test_df


def add_vertex_risk_feature(train_df, test_df):
    """
    Compute vertex_risk_factor ONLY from training data to prevent leakage.
    Maps each (x_norm, y_norm) grid position to its historical conflict rate
    as observed in the training set.
    """
    print("\n[2/4] Computing vertex_risk_factor from training partition only...")

    # Use rounded normalised coordinates as a proxy for vertex identity
    # (original collision_vertex was dropped to prevent leakage)
    for part in [train_df, test_df]:
        part["_grid_key"] = (
            part["x_norm"].round(3).astype(str) + "_" +
            part["y_norm"].round(3).astype(str)
        )

    risk_map = (
        train_df.groupby("_grid_key")[TARGET_COL].mean()
    ).to_dict()

    train_df["vertex_risk_factor"] = train_df["_grid_key"].map(risk_map).fillna(0)
    test_df["vertex_risk_factor"]  = test_df["_grid_key"].map(risk_map).fillna(0)

    train_df.drop(columns=["_grid_key"], inplace=True)
    test_df.drop(columns=["_grid_key"], inplace=True)

    return train_df, test_df


def prepare_features(train_df, test_df):
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COL].astype(int)
    return X_train, y_train, X_test, y_test, feature_cols


def train_model(X_train, y_train):
    print("\n[3/4] Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    print("      Training complete.")
    return model


def evaluate(model, X_test, y_test, feature_cols):
    print("\n[4/4] Evaluating...")
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_pred_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc   = auc(rec_curve, prec_curve)
    cm       = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 60)
    print("        EVALUATION RESULTS  --  processed dataset")
    print("=" * 60)
    print(f"\n  Overall Accuracy : {accuracy*100:.2f}%")
    print(f"  ROC-AUC Score    : {roc_auc:.4f}")
    print(f"  PR-AUC Score     : {pr_auc:.4f}")
    print("\n  Confusion Matrix:")
    print(f"  {'':22} {'Pred: 0':^12} {'Pred: 1':^12}")
    print(f"  {'Actual: 0 (Normal)':22} TN={tn:<10} FP={fp:<10}")
    print(f"  {'Actual: 1 (Risk)':22} FN={fn:<10} TP={tp:<10}")
    print("\n  Per-class Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Normal Flow (0)", "Conflict Risk (1)"],
        digits=4
    ))
    print("  Top 10 Feature Importances:")
    ranked = sorted(zip(model.feature_importances_, feature_cols), reverse=True)
    for score, name in ranked[:10]:
        bar = "#" * int(score * 50)
        print(f"  {name:<30} {score:.4f}  {bar}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"Error: '{CSV_PATH}' not found.", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Evaluation pipeline for processed_congestion_dataset.csv")
    train_df, test_df = load_and_split(CSV_PATH)
    train_df, test_df = add_vertex_risk_feature(train_df, test_df)
    X_train, y_train, X_test, y_test, feature_cols = prepare_features(train_df, test_df)

    if X_train.empty or X_test.empty:
        print("Error: Train or test split is empty.", file=sys.stderr)
        sys.exit(1)

    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test, feature_cols)
