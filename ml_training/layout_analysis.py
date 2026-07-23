import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
    Phân tích chuyên sâu hiện tượng 'layout memorization' và xuất kết quả hoàn toàn dưới dạng Text Report.
    """
    print("\n" + "="*50)
    print("      TEXT REPORT: SPATIAL LAYOUT MEMORIZATION       ")
    print("="*50)
    
    if not os.path.exists(spatial_model_path):
        print(f"[Lỗi] Không tìm thấy model tại {spatial_model_path}")
        return

    if not os.path.exists(processed_data_path):
        print(f"[Lỗi] Không tìm thấy processed dataset tại {processed_data_path}")
        return
        
    # 1. Đọc processed dataset
    df_proc = pd.read_csv(processed_data_path)
    df_proc = df_proc.dropna(subset=["target_future_conflict"])
    df_proc["target_future_conflict"] = df_proc["target_future_conflict"].astype(int)

    # Tìm file raw để map tọa độ x, y
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
        print("[Lỗi] Không tìm thấy file congestion_dataset.csv gốc để lấy tọa độ x, y!")
        return
    
    if "x" not in df_proc.columns or "y" not in df_proc.columns:
        print("[Lỗi] Không thể gán được cột 'x' và 'y'!")
        return

    # 2. Tạo Ground Truth Map
    heat = df_proc.groupby(["y", "x"])["target_future_conflict"].mean().reset_index()
    heatmap_target = heat.pivot(index="y", columns="x", values="target_future_conflict")

    # 3. Tạo Model Prediction Map
    package = joblib.load(spatial_model_path)
    model = package["model"]
    features = package["features"]

    available_features = [col for col in features if col in df_proc.columns]
    X_proc = df_proc[available_features]
    df_proc["predicted_prob"] = model.predict_proba(X_proc)[:, 1]

    pred_heat = df_proc.groupby(["y", "x"])["predicted_prob"].mean().reset_index()
    heatmap_pred = pred_heat.pivot(index="y", columns="x", values="predicted_prob")

    # 4. Căn chỉnh ma trận để so khớp
    aligned_target, aligned_pred = heatmap_target.align(heatmap_pred, join="inner")
    
    target_stacked = aligned_target.stack()
    pred_stacked = aligned_pred.stack()
    
    valid_idx = ~(target_stacked.isna() | pred_stacked.isna())
    t_vals = target_stacked[valid_idx]
    p_vals = pred_stacked[valid_idx]
    cell_count = len(t_vals)

    # 5. Tính toán các metric thống kê & Entropy
    def binary_entropy(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return - (p * np.log2(p) + (1 - p) * np.log2(1 - p))

    entropy_df = df_proc.groupby(["y", "x"])["target_future_conflict"].agg(lambda p: binary_entropy(p.mean())).reset_index(name="entropy")
    mean_entropy = entropy_df["entropy"].mean()
    
    pearson_corr, _ = pearsonr(t_vals, p_vals)
    spearman_corr, _ = spearmanr(t_vals, p_vals)

    # 6. In kết quả dạng Text hoàn toàn
    print(f"[1] Tổng số ô kho (Warehouse Cells) được đánh giá : {cell_count}")
    print(f"[2] Pearson Correlation  (Target vs Prediction)  : {pearson_corr:.4f}")
    print(f"[3] Spearman Correlation (Target vs Prediction)  : {spearman_corr:.4f}")
    print(f"[4] Mean Binary Entropy H(target | x,y)          : {mean_entropy:.4f}")
    
    print("\n--- Đánh giá nhanh tính chất Layout Memorization ---")
    if mean_entropy < 0.3:
        print("-> Nhận định: Entropy trung bình rất thấp (gần 0). Điều này chứng minh không gian kho mang tính quyết định (deterministic) cao theo vị trí, giải thích vì sao mô hình Spatial đạt hiệu năng rất tốt chỉ nhờ ghi nhớ layout.")
    else:
        print("-> Nhận định: Entropy ở mức trung bình/cao, cho thấy sự biến động lớn tại các ô, mô hình cần kết hợp thêm yếu tố thời gian (temporal dynamics).")
    print("="*50)

if __name__ == "__main__":
    # 2. Chạy phân tích chuyên sâu Layout Memorization & Entropy (6 figures)
    spatial_path = os.path.join(MODEL_DIR, "spatial_model.pkl")
    
    evaluate_layout_memorization(
        spatial_model_path=spatial_path, 
        processed_data_path=DATA_PATH
    )