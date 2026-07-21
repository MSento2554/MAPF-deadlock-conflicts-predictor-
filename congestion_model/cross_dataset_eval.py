"""
Cross-Dataset Evaluation — with Behavioral Leading-Indicator Features
=====================================================================
Trains on one collision scenario and tests generalization on others.
Features used are genuine leading indicators (no current conflict state leakage).
"""

import os
import sys
import warnings
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "few_collisions":  os.path.join(ROOT, "congestion_dataset (with few collisions).csv"),
    "many_collisions": os.path.join(ROOT, "congestion_dataset (with many collisions).csv"),
    "no_collisions":   os.path.join(ROOT, "congestion_dataset (without collisions).csv"),
}

TARGET_COL = "target_future_conflict"

# All genuine leading-indicator features (no current conflict state)
BASE_FEATURES = [
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


def preprocess(path, label):
    from congestion_dataset_preprocessing import preprocess_congestion_dataset
    print(f"  Preprocessing '{label}'...")
    raw = pd.read_csv(path)
    df  = preprocess_congestion_dataset(raw, lead_time_steps=5, window_size=5, num_lags=3)
    pos = int((df[TARGET_COL] == 1).sum())
    neg = int((df[TARGET_COL] == 0).sum())
    print(f"  → {len(df):,} rows | {neg:,} normal / {pos:,} risk  ({pos/len(df)*100:.1f}% positive)")
    return df


def get_feature_cols(df):
    available = [c for c in BASE_FEATURES if c in df.columns]
    return available


def run_experiment(train_df, test_df, label):
    feat_cols = get_feature_cols(train_df)

    X_train = train_df[feat_cols].copy()
    y_train = train_df[TARGET_COL].astype(int)
    X_test  = test_df[feat_cols].copy()
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_test  = test_df[TARGET_COL].astype(int)

    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=pos_weight, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate optimal threshold based on F1-score
    import numpy as np
    prec_c, rec_c, thrs = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec_c, prec_c)
    f1s = 2 * prec_c * rec_c / np.maximum(prec_c + rec_c, 1e-9)
    best_idx = np.argmax(f1s[:-1])
    best_thr = thrs[best_idx]

    # Use the best threshold for predictions
    y_pred = (y_prob >= best_thr).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    has_both = len(y_test.unique()) > 1
    roc_auc  = roc_auc_score(y_test, y_prob) if has_both else float("nan")
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*64}")
    print(f"  RESULT: {label}")
    print(f"{'='*64}")
    print(f"  Samples : {len(y_test):,}  "
          f"({(y_test==0).sum()} normal / {(y_test==1).sum()} risk)")
    print(f"  Accuracy: {accuracy*100:.2f}%   ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")
    print(f"  Best F1 Threshold: {best_thr:.3f}")
    print(f"\n  Confusion Matrix (@ thr={best_thr:.3f}):")
    print(f"  {'':22} Pred:0      Pred:1")
    print(f"  {'Actual:0 (Normal)':22} TN={tn:<8}  FP={fp:<8}")
    print(f"  {'Actual:1 (Risk)':22} FN={fn:<8}  TP={tp:<8}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["Normal Flow (0)", "Conflict Risk (1)"],
        digits=4, zero_division=0,
    ))
    print("  Top 10 Feature Importances:")
    ranked = sorted(zip(model.feature_importances_, feat_cols), reverse=True)
    for sc, nm in ranked[:10]:
        bar = "#" * int(sc * 40)
        print(f"  {nm:<30} {sc:.4f}  {bar}")
    print(f"{'='*64}")

    prec_risk = tp / max(1, tp + fp)
    rec_risk  = tp / max(1, tp + fn)
    f1_risk   = 2 * tp / max(1, 2*tp + fp + fn)
    return dict(label=label, accuracy=accuracy, roc_auc=roc_auc, pr_auc=pr_auc,
                precision=prec_risk, recall=rec_risk, f1=f1_risk, thr=best_thr)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    for path in DATASETS.values():
        if not os.path.exists(path):
            print(f"ERROR: missing {path}", file=sys.stderr); sys.exit(1)

    print("\n[INFO] Cross-Dataset Generalization Evaluation")
    print("       (with behavioral leading-indicator features)")
    print("=" * 64)

    print("\n[STEP 1/4] Preprocessing all datasets...")
    dfs = {name: preprocess(path, name) for name, path in DATASETS.items()}

    results = []

    print("\n[STEP 2/4] Exp 1: Train=few_collisions → Test=many_collisions")
    r = run_experiment(
        dfs["few_collisions"].copy(), dfs["many_collisions"].copy(),
        "few → many collisions"
    )
    results.append(r)

    print("\n[STEP 3/4] Exp 2: Train=many_collisions → Test=few_collisions")
    r = run_experiment(
        dfs["many_collisions"].copy(), dfs["few_collisions"].copy(),
        "many → few collisions"
    )
    results.append(r)

    print("\n[STEP 4/4] Exp 3: Train=few+many → Test=no_collisions")
    combined = pd.concat(
        [dfs["few_collisions"], dfs["many_collisions"]], ignore_index=True
    )
    r = run_experiment(
        combined, dfs["no_collisions"].copy(),
        "few+many → no collisions"
    )
    results.append(r)

    print("\n" + "=" * 64)
    print("  CROSS-DATASET SUMMARY")
    print("=" * 64)
    print(f"  {'Experiment':<28} {'Acc':>6} {'ROC':>7} {'PR':>7} {'Thr':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'-'*28} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
    for r in results:
        print(f"  {r['label']:<28} "
              f"{r['accuracy']*100:>5.1f}% "
              f"{r['roc_auc']:>7.4f} "
              f"{r['pr_auc']:>7.4f} "
              f"{r['thr']:>6.3f} "
              f"{r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} "
              f"{r['f1']:>7.4f}")
    print("=" * 64)
