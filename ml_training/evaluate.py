import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

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


def evaluate_single_model(
    model_name, model_package_path, test_df, target_col="target_future_conflict"
):
    """Evaluate a single model (Spatial or Temporal) on the test set using standard reporting metrics."""
    if not os.path.exists(model_package_path):
        print(f"[{model_name}] Model package not found at {model_package_path}")
        return None

    package = joblib.load(model_package_path)

    model = package["model"]
    threshold = package.get("threshold", 0.5)
    features = package["features"]
    cv_results = package.get("cv_results", None)

    # Filter valid features present in the test set
    available_features = [col for col in features if col in test_df.columns]
    X_test = test_df[available_features]
    y_test = test_df[target_col].astype(int)

    # Predict probabilities & apply threshold
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    # Compute test metrics
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    roc = roc_auc_score(y_test, prob)
    pr = average_precision_score(y_test, prob)

    print(f"\n=== {model_name} Standard Evaluation ===")
    print(f"Optimal Threshold : {threshold:.3f}")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Precision         : {prec:.4f}")
    print(f"Recall            : {rec:.4f}")
    print(f"F1-Score          : {f1:.4f}")
    print(f"ROC-AUC           : {roc:.4f}")
    print(f"PR-AUC            : {pr:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pred, zero_division=0))

    # Print CV Summary if available in package
    if cv_results:
        print(f"\n--- {model_name} 5-Fold TimeSeries CV Summary ---")
        for key in ["acc", "precision", "recall", "f1", "roc", "pr"]:
            if key in cv_results[0]:
                scores = [fold[key] for fold in cv_results]
                print(f"{key.upper():10s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # --- Visualizations ---
    color_map = "Blues" if "spatial" in model_name.lower() else "Oranges"
    file_suffix = "spatial" if "spatial" in model_name.lower() else "temporal"

    # 1. Normalized Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, pred, normalize="true")
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap=color_map,
        xticklabels=["Normal", "Conflict"],
        yticklabels=["Normal", "Conflict"],
    )
    plt.title(f"{model_name} - Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(
        os.path.join(FIGURES_DIR, f"confusion_matrix_{file_suffix}.png"),
        bbox_inches="tight",
    )
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr,
        color="darkblue" if "spatial" in file_suffix else "darkorange",
        linewidth=2,
        label=f"ROC-AUC = {roc:.3f}",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(
        os.path.join(FIGURES_DIR, f"roc_curve_{file_suffix}.png"),
        bbox_inches="tight",
    )
    plt.close()

    # 3. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, prob)
    plt.figure(figsize=(6, 5))
    plt.plot(
        recall_vals,
        precision_vals,
        color="purple" if "spatial" in file_suffix else "green",
        linewidth=2,
        label=f"PR-AUC = {pr:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(
        os.path.join(FIGURES_DIR, f"pr_curve_{file_suffix}.png"),
        bbox_inches="tight",
    )
    plt.close()

    return {
        "Threshold": threshold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc,
        "PR-AUC": pr,
    }


def evaluate_models():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # 1. Clean target and sort chronologically to match training pipeline
    df = df.dropna(subset=["target_future_conflict"])
    df["target_future_conflict"] = df["target_future_conflict"].astype(int)
    df = df.sort_values(["t", "robot_id"]).reset_index(drop=True)

    # 2. Strict time-based split (80/20 rule)
    split_tick = df["t"].quantile(0.8)
    test_df = df[df["t"] > split_tick]

    print(f"Total Test Samples: {len(test_df)} (Tick > {split_tick})")

    # 3. Evaluate Spatial Model
    spatial_path = os.path.join(MODEL_DIR, "spatial_model.pkl")
    spatial_metrics = evaluate_single_model("Spatial Model", spatial_path, test_df)

    # 4. Evaluate Temporal Model
    temporal_path = os.path.join(MODEL_DIR, "temporal_model.pkl")
    temporal_metrics = evaluate_single_model("Temporal Model", temporal_path, test_df)

    # 5. Summarize test performance and generate comparison table
    if spatial_metrics and temporal_metrics:
        summary_df = pd.DataFrame([
            {"Model": "Spatial Model", **spatial_metrics},
            {"Model": "Temporal Model", **temporal_metrics},
        ])
        print("\n==================================================")
        print("          FINAL TEST SET PERFORMANCE SUMMARY      ")
        print("==================================================")
        print(summary_df.to_string(index=False))
        print("==================================================")

    print(f"\nEvaluation completed successfully! All visual figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    # Execute standard model evaluation pipeline (Confusion Matrices, ROC, PR curves, etc.)
    evaluate_models()
    
