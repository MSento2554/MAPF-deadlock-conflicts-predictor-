import os
import sys
import numpy as np
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

# Import the shared preprocessing pipeline — no duplicate logic here
from congestion_dataset_preprocessing import preprocess_congestion_dataset

# ── Feature configuration ─────────────────────────────────────────────────────
TARGET_COL = "target_future_conflict"

# Columns excluded from model input (metadata / target)
EXCLUDE_COLS = {"t", "robot_id", TARGET_COL}

# Genuine leading-indicator behavioral features (no current-conflict state)
# vertex_risk_factor is added separately after the train/test split (train-only)
FEATURE_COLS = [
    # Robot state
    "state_AVAILABLE", "state_IN_PROGRESS", "is_carrying",
    # Spatial position
    "x_norm", "y_norm",
    # Position stability
    "steps_stationary", "stationary_ratio",
    # System-wide congestion pressure
    "neighbour_static_count",
    # Rolling density (absolute)
    "local_density_3_smooth", "local_density_5_smooth",
    # Scale-invariant density (z-score, peak, acceleration)
    "density_3_zscore",  "density_5_zscore",
    "density_3_peak",    "density_5_peak",
    "density_3_accel",   "density_5_accel",
    # Density trend and spread
    "density_3_delta",   "density_5_delta",
    "density_spread",
    # Density lag snapshots
    "density_3_lag_1", "density_3_lag_2", "density_3_lag_3",
    "density_5_lag_1", "density_5_lag_2", "density_5_lag_3",
]
# ─────────────────────────────────────────────────────────────────────────────


def split_train_test(df: pd.DataFrame):
    """
    Time-based 80/20 split — no shuffling to preserve temporal order and
    prevent future data leaking into the training window.
    """
    split_tick = df["t"].max() * 0.8
    train_df = df[df["t"] <= split_tick].copy()
    test_df  = df[df["t"] >  split_tick].copy()
    print(f"      Train: {len(train_df):,} samples (t ≤ {split_tick:.0f})")
    print(f"      Test : {len(test_df):,} samples  (t >  {split_tick:.0f})")
    return train_df, test_df


def add_vertex_risk_factor(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Computes a spatial conflict-risk score per grid cell using ONLY the training
    partition. Mapping to the test set without re-fitting avoids target leakage.
    """
    for part in [train_df, test_df]:
        part["_gk"] = (
            part["x_norm"].round(3).astype(str) + "_" +
            part["y_norm"].round(3).astype(str)
        )

    risk_map = train_df.groupby("_gk")[TARGET_COL].mean().to_dict()
    train_df["vertex_risk_factor"] = train_df["_gk"].map(risk_map).fillna(0)
    test_df["vertex_risk_factor"]  = test_df["_gk"].map(risk_map).fillna(0)

    for part in [train_df, test_df]:
        part.drop(columns=["_gk"], inplace=True)


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Selects the final feature columns for training and testing."""
    feat_cols = [c for c in FEATURE_COLS + ["vertex_risk_factor"]
                 if c in train_df.columns]

    X_train = train_df[feat_cols]
    y_train = train_df[TARGET_COL].astype(int)
    X_test  = test_df[feat_cols].reindex(columns=X_train.columns, fill_value=0)
    y_test  = test_df[TARGET_COL].astype(int)

    return X_train, y_train, X_test, y_test, feat_cols


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Trains an XGBoost classifier with scale_pos_weight to handle any remaining
    class imbalance, and regularisation (subsample / colsample_bytree) to reduce
    overfitting on small datasets.
    """
    num_negative = (y_train == 0).sum()
    num_positive = (y_train == 1).sum()
    pos_weight   = num_negative / max(1, num_positive)

    print("\n--- MODEL METADATA ---")
    print(f"  Training vectors : {len(X_train):,}")
    print(f"  Class balance    : {num_negative:,} normal  /  {num_positive:,} risk")
    print(f"  scale_pos_weight : {pos_weight:.2f}")
    print(f"  Features         : {len(X_train.columns)}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Done.")
    return model


def evaluate(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list,
) -> dict:
    """
    Full evaluation suite:
      - Default threshold (0.5) metrics
      - Threshold sweep table
      - Best-F1 threshold metrics with confusion matrix
      - Top 10 feature importances
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = model.predict(X_test)

    # ── Core scores ──────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_pred_proba)
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(rec_curve, prec_curve)

    # ── Best-F1 threshold ─────────────────────────────────────────────────────
    f1_scores = (
        2 * prec_curve * rec_curve /
        np.maximum(prec_curve + rec_curve, 1e-9)
    )
    best_idx = int(np.argmax(f1_scores[:-1]))
    best_thr = float(thresholds[best_idx])
    y_pred_best = (y_pred_proba >= best_thr).astype(int)

    cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 60)
    print("              EVALUATION METRICS")
    print("=" * 60)
    print(f"\n  [Default threshold = 0.5]")
    print(f"  Accuracy  : {accuracy*100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")

    print(f"\n  [Threshold sweep]")
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'-'*42}")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, best_thr]:
        yp = (y_pred_proba >= thr).astype(int)
        if yp.sum() == 0:
            continue
        p = ((yp == 1) & (y_test == 1)).sum() / yp.sum()
        r = ((yp == 1) & (y_test == 1)).sum() / max(1, (y_test == 1).sum())
        f = 2 * p * r / max(p + r, 1e-9)
        tag = " ← default" if abs(thr - 0.5) < 0.001 else (
              " ← BEST F1" if abs(thr - best_thr) < 0.001 else "")
        print(f"  {thr:>10.3f}  {p:>10.4f}  {r:>8.4f}  {f:>8.4f}{tag}")

    print(f"\n  [Best threshold = {best_thr:.3f}]")
    print(f"  Confusion Matrix:")
    print(f"  {'':20}  Pred: 0     Pred: 1")
    print(f"  {'Actual: 0 (Normal)':20}  TN={tn:<8}  FP={fp:<8}")
    print(f"  {'Actual: 1 (Risk)':20}  FN={fn:<8}  TP={tp:<8}")
    print()
    print(classification_report(
        y_test, y_pred_best,
        target_names=["Normal Flow (0)", "Deadlock Risk (1)"],
        digits=4,
        zero_division=0,
    ))

    print("  Top 10 Feature Importances:")
    ranked = sorted(zip(model.feature_importances_, feature_cols), reverse=True)
    for score, name in ranked[:10]:
        bar = "#" * int(score * 45)
        print(f"  {name:<30}  {score:.4f}  {bar}")

    print("\n" + "=" * 60)

    return {
        "accuracy":  accuracy,
        "roc_auc":   roc_auc,
        "pr_auc":    pr_auc,
        "best_thr":  best_thr,
        "best_f1":   float(f1_scores[best_idx]),
        "precision": float(prec_curve[best_idx]),
        "recall":    float(rec_curve[best_idx]),
    }


# =====================================================================
# STANDALONE EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    # ── Config ────────────────────────────────────────────────────────
    # Point this at whichever raw dataset you want to evaluate.
    # The file path is relative to this script's directory.
    csv_filename = "congestion_dataset.csv"
    # Prediction horizon: how many ticks ahead to predict conflict
    LEAD_TIME_STEPS = 5
    # Rolling window size for smoothed/lag features
    WINDOW_SIZE = 5
    # Number of lag steps for density features
    NUM_LAGS = 3
    # ─────────────────────────────────────────────────────────────────

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, csv_filename)

    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found.", file=sys.stderr)
        print("Ensure the dataset CSV is in the same directory as this script.",
              file=sys.stderr)
        sys.exit(1)

    print(f"\n[1/4] Loading dataset: '{csv_filename}'")
    raw_df = pd.read_csv(csv_path)
    print(f"      Raw shape: {raw_df.shape}")

    print("\n[2/4] Running preprocessing pipeline...")
    df = preprocess_congestion_dataset(
        raw_df,
        lead_time_steps=LEAD_TIME_STEPS,
        window_size=WINDOW_SIZE,
        num_lags=NUM_LAGS,
    )
    pos = (df[TARGET_COL] == 1).sum()
    neg = (df[TARGET_COL] == 0).sum()
    print(f"\n      Processed shape: {df.shape}")
    print(f"      Target balance : {neg:,} normal / {pos:,} risk "
          f"({pos / len(df) * 100:.1f}% positive)")

    print("\n[3/4] Splitting and preparing features...")
    train_df, test_df = split_train_test(df)

    if train_df.empty or test_df.empty:
        print("Error: train or test split is empty — not enough time steps.",
              file=sys.stderr)
        sys.exit(1)

    add_vertex_risk_factor(train_df, test_df)
    X_train, y_train, X_test, y_test, feat_cols = prepare_features(train_df, test_df)

    print(f"\n[4/4] Training and evaluating...")
    model = train_model(X_train, y_train)
    results = evaluate(model, X_test, y_test, feat_cols)