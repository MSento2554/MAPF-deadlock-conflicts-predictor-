import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

def run_preprocessing_pipeline(raw_df, lead_time_steps=10, num_lags=5):
    """
    Executes the complete preprocessing pipeline for predicting warehouse robot deadlocks.
    Ensures no data leakage occurs between training and validation stages.
    """
    print("\n[1/8] Sorting chronologically per robot...")
    df = raw_df.sort_values(by=["robot_id", "t"]).reset_index(drop=True)
    
    print("[2/8] Engineering dynamic kinematics (Speed/Velocity)...")
    grouped = df.groupby("robot_id")
    df["dx"] = grouped["x"].diff().fillna(0)
    df["dy"] = grouped["y"].diff().fillna(0)
    df["speed"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    
    print("[3/8] Creating predictive target label via forward-lookahead...")
    # Label a row as '1' if the robot hits a deadlock anytime within the upcoming window
    df['will_deadlock'] = (
        df.iloc[::-1]
        .groupby('robot_id')['is_deadlocked']
        .rolling(window=lead_time_steps, min_periods=1)
        .max()
        .iloc[::-1]
        .reset_index(level=0, drop=True)
    ).astype(int)
    
    print("[4/8] Categorical encoding of robot tracking states...")
    # Safely extract states dynamically
    if "state" in df.columns:
        df = pd.get_dummies(df, columns=["state"], prefix="state", dtype=int)
    
    # Binary status flag for payload presence
    if "held_item" in df.columns:
        df["is_carrying"] = df["held_item"].notna().astype(int)
    else:
        df["is_carrying"] = 0
    
    print("[5/8] Performing time-based splitting to prevent temporal data leakage...")
    # Use the chronologically first 80% of frames for training, final 20% for validation testing
    split_tick = df['t'].max() * 0.8
    train_df = df[df['t'] <= split_tick].copy()
    test_df = df[df['t'] > split_tick].copy()
    
    print("[6/8] Mapping spatial hot-spots (Vertex Historical Risk Weights)...")
    if "collision_vertex" in df.columns:
        # Evaluate localized risk mapping strictly using the training matrix
        vertex_risk = train_df.groupby("collision_vertex")["is_stuck"].mean().to_dict()
        train_df["vertex_historical_risk"] = train_df["collision_vertex"].map(vertex_risk).fillna(0)
        test_df["vertex_historical_risk"] = test_df["collision_vertex"].map(vertex_risk).fillna(0)
    else:
        train_df["vertex_historical_risk"] = 0
        test_df["vertex_historical_risk"] = 0
    
    print("[7/8] Generating time-series lag sequences...")
    # Columns to look back on dynamically
    features_to_lag = ["speed", "stuck_duration"]
    for density_col in ["local_density_3", "local_density_5", "local_density"]:
        if density_col in df.columns:
            features_to_lag.append(density_col)
            
    for partition in [train_df, test_df]:
        for lag in range(1, num_lags + 1):
            for col in features_to_lag:
                partition[f"{col}_lag_{lag}"] = partition.groupby("robot_id")[col].shift(lag)
                
    print("[8/8] Cleaning tracking indicators and isolating features...")
    # Drop rows at the beginning of timelines that lack sufficient lookback histories
    train_clean = train_df.dropna().reset_index(drop=True)
    test_clean = test_df.dropna().reset_index(drop=True)
    
    # Unusable columns to remove entirely before training
    exclude_cols = [
        "t", "robot_id", "x", "y", "held_item", "collision_vertex", 
        "collision_edge", "is_stuck", "is_deadlocked", "will_deadlock"
    ]
    
    feature_cols = [col for col in train_clean.columns if col not in exclude_cols]
    
    X_train = train_clean[feature_cols]
    y_train = train_clean["will_deadlock"]
    X_test = test_clean[feature_cols]
    y_test = test_clean["will_deadlock"]
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate_baseline(X_train, y_train, X_test, y_test):
    """
    Trains an XGBoost model adjusting for class imbalances and provides analytical diagnostics.
    """
    num_negative = (y_train == 0).sum()
    num_positive = (y_train == 1).sum()
    pos_weight = num_negative / max(1, num_positive)
    
    print("\n--- MODEL METADATA ---")
    print(f"Dataset Size:   {X_train.shape[0]} training vectors | {X_test.shape[0]} testing vectors")
    print(f"Class Balance:  {num_negative} normal steps / {num_positive} risk frames")
    print(f"Weight Tuning:  scale_pos_weight optimized to {pos_weight:.2f}")
    
    # Initialize light baseline classifier configuration
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=pos_weight, 
        eval_metric="logloss",
        random_state=42
    )
    
    print("\nTraining Model...")
    model.fit(X_train, y_train)
    
    # Generate diagnostic vectors
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    print("\n================ EVALUATION METRICS ================")
    print(classification_report(y_test, y_pred, target_names=["Normal Flow", "Deadlock Risk"]))
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"ROC-AUC Score:      {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"PR-AUC Score:       {pr_auc:.4f}")
    print("====================================================")
    
    print("\nTop 5 Feature Importances:")
    importance = model.feature_importances_
    feature_ranking = sorted(zip(importance, X_train.columns), reverse=True)
    for score, name in feature_ranking[:5]:
        print(f" - {name}: {score:.4f}")
        
    return model

# =====================================================================
# STANDALONE EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    # Define file configurations
    csv_filename = "congestion_dataset (with few collisions).csv"
    
    # Verify local file presence
    if not os.path.exists(csv_filename):
        print(f"Error: Could not find '{csv_filename}' in the active folder directory.", file=sys.stderr)
        print("Please ensure your tracking simulator logs match this designation.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found dataset log: Loading '{csv_filename}'...")
    raw_df = pd.read_csv(csv_filename)
    
    # Run pipeline commands (Target: Predict anomaly up to 10 ticks ahead, look back 5 steps)
    X_train, y_train, X_test, y_test = run_preprocessing_pipeline(
        raw_df, 
        lead_time_steps=10, 
        num_lags=5
    )
    
    # Check if data reduction left the vectors completely empty
    if X_train.empty or X_test.empty:
        print("\nError: The resulting splits contain no vectors. Verify that the csv contains sufficient timeline intervals.", file=sys.stderr)
        sys.exit(1)
        
    # Launch training process
    model = train_and_evaluate_baseline(X_train, y_train, X_test, y_test)