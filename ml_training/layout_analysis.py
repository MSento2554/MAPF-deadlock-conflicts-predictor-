import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# Directory configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    os.path.dirname(current_dir),
    "congestion_model",
    "processed_congestion_dataset.csv",
)
MODEL_DIR = os.path.join(current_dir, "models")
FIGURES_DIR = os.path.join(current_dir, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

def evaluate_layout_memorization(spatial_model_path, processed_data_path):
    """
    In-depth analysis of spatial layout characteristics and model predictions (Text Report).
    """
    print("\n" + "="*50)
    print("      TEXT REPORT: SPATIAL LAYOUT MEMORIZATION       ")
    print("="*50)
    
    if not os.path.exists(spatial_model_path):
        print(f"[Error] Model not found at {spatial_model_path}")
        return

    if not os.path.exists(processed_data_path):
        print(f"[Error] Processed dataset not found at {processed_data_path}")
        return
        
    # 1. Load processed dataset
    df_proc = pd.read_csv(processed_data_path)
    df_proc = df_proc.dropna(subset=["target_future_conflict"])
    df_proc["target_future_conflict"] = df_proc["target_future_conflict"].astype(int)

    # Locate raw dataset to map (x, y) coordinates
    possible_paths = [
        os.path.join(os.path.dirname(processed_data_path), "congestion_dataset.csv"),
        os.path.join(os.path.dirname(os.path.dirname(processed_data_path)), "congestion_dataset.csv"),
        "congestion_dataset.csv",
        "../congestion_dataset.csv"
    ]
    
    raw_data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            raw_data_path = p
            break

    if raw_data_path:
        df_raw = pd.read_csv(raw_data_path)
        if "robot_id" in df_proc.columns and "t" in df_proc.columns and "x" in df_raw.columns and "y" in df_raw.columns:
            if "x" in df_proc.columns: df_proc = df_proc.drop(columns=["x"])
            if "y" in df_proc.columns: df_proc = df_proc.drop(columns=["y"])
            df_proc = pd.merge(df_proc, df_raw[["robot_id", "t", "x", "y"]], on=["robot_id", "t"], how="left")
    else:
        print("[Error] Raw dataset 'congestion_dataset.csv' not found for coordinate mapping!")
        return
    
    if "x" not in df_proc.columns or "y" not in df_proc.columns:
        print("[Error] Failed to assign 'x' and 'y' coordinates!")
        return

    # 2. Build Ground Truth Map
    heat = df_proc.groupby(["y", "x"])["target_future_conflict"].mean().reset_index()
    heatmap_target = heat.pivot(index="y", columns="x", values="target_future_conflict")

    # 3. Build Model Prediction Map
    package = joblib.load(spatial_model_path)
    model = package["model"]
    features = package["features"]

    available_features = [col for col in features if col in df_proc.columns]
    X_proc = df_proc[available_features]
    df_proc["predicted_prob"] = model.predict_proba(X_proc)[:, 1]

    pred_heat = df_proc.groupby(["y", "x"])["predicted_prob"].mean().reset_index()
    heatmap_pred = pred_heat.pivot(index="y", columns="x", values="predicted_prob")

    # 4. Align matrices for comparison
    aligned_target, aligned_pred = heatmap_target.align(heatmap_pred, join="inner")
    
    target_stacked = aligned_target.stack()
    pred_stacked = aligned_pred.stack()
    
    valid_idx = ~(target_stacked.isna() | pred_stacked.isna())
    t_vals = target_stacked[valid_idx]
    p_vals = pred_stacked[valid_idx]
    cell_count = len(t_vals)

    # 5. Compute statistical metrics & Entropy
    def binary_entropy(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return - (p * np.log2(p) + (1 - p) * np.log2(1 - p))

    entropy_df = df_proc.groupby(["y", "x"])["target_future_conflict"].agg(lambda p: binary_entropy(p.mean())).reset_index(name="entropy")
    mean_entropy = entropy_df["entropy"].mean()
    
    pearson_corr, _ = pearsonr(t_vals, p_vals)
    spearman_corr, _ = spearmanr(t_vals, p_vals)

    # 6. Output text report
    print(f"[1] Evaluated warehouse cells count       : {cell_count}")
    print(f"[2] Pearson Correlation (Target vs Pred)  : {pearson_corr:.4f}")
    print(f"[3] Spearman Correlation (Target vs Pred) : {spearman_corr:.4f}")
    print(f"[4] Mean Binary Entropy H(target | x,y)   : {mean_entropy:.4f}")
    
    print("\n--- Layout Memorization Insights ---")
    if mean_entropy < 0.3:
        print("-> Insight: Low mean entropy suggests that future conflicts are highly deterministic with respect to spatial location, partially explaining the strong performance of the Spatial Model.")
    else:
        print("-> Insight: Moderate/High entropy indicates higher variance across cells, implying that temporal dynamics play a more substantial role.")
    print("="*50)

if __name__ == "__main__":
    spatial_path = os.path.join(MODEL_DIR, "spatial_model.pkl")
    
    evaluate_layout_memorization(
        spatial_model_path=spatial_path, 
        processed_data_path=DATA_PATH
    )