import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
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


def evaluate_models():
    # Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # Generate target label
    df["is_congested"] = (
        df["local_density_5_smooth"] > 2.5
    ).astype(int)

    y = df["is_congested"]

    # Feature sets
    spatial_features = [
        "x_norm",
        "y_norm",
        "local_density_3_smooth",
        "density_spread",
        "density_3_zscore",
        "density_5_zscore",
    ]

    temporal_features = [
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

    spatial_cols = [
        col for col in spatial_features
        if col in df.columns
    ]

    temporal_cols = [
        col for col in temporal_features
        if col in df.columns
    ]

    # Shared test split
    _, X_spatial_test, _, y_test = train_test_split(
        df[spatial_cols],
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    _, X_temporal_test, _, _ = train_test_split(
        df[temporal_cols],
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Evaluate Spatial model
    spatial_model_path = os.path.join(
        MODEL_DIR,
        "spatial_model.pkl",
    )

    if os.path.exists(spatial_model_path):
        print("=== Spatial Model Evaluation ===")

        spatial_model = joblib.load(spatial_model_path)

        y_pred_spatial = spatial_model.predict(X_spatial_test)
        y_proba_spatial = spatial_model.predict_proba(
            X_spatial_test
        )[:, 1]

        print("Accuracy:", accuracy_score(y_test, y_pred_spatial))
        print("ROC-AUC:", roc_auc_score(y_test, y_proba_spatial))
        print(classification_report(y_test, y_pred_spatial))

        # Confusion matrix
        plt.figure(figsize=(6, 5))

        cm_spatial = confusion_matrix(
            y_test,
            y_pred_spatial,
        )

        sns.heatmap(
            cm_spatial,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Free", "Congested"],
            yticklabels=["Free", "Congested"],
        )

        plt.title("Spatial Model - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        plt.savefig(
            os.path.join(
                FIGURES_DIR,
                "confusion_matrix_spatial.png",
            ),
            bbox_inches="tight",
        )

        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(
            y_test,
            y_proba_spatial,
        )

        auc_score = roc_auc_score(
            y_test,
            y_proba_spatial,
        )

        plt.figure(figsize=(6, 5))

        plt.plot(
            fpr,
            tpr,
            color="blue",
            linewidth=2,
            label=f"AUC = {auc_score:.3f}",
        )

        plt.plot(
            [0, 1],
            [0, 1],
            "k--",
            label="Random Classifier",
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Spatial Model - ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        plt.savefig(
            os.path.join(
                FIGURES_DIR,
                "roc_curve_spatial.png",
            ),
            bbox_inches="tight",
        )

        plt.close()

    else:
        print(f"Spatial model not found: {spatial_model_path}")

    # Evaluate Temporal model
    temporal_model_path = os.path.join(
        MODEL_DIR,
        "temporal_model.pkl",
    )

    if os.path.exists(temporal_model_path):
        print("\n=== Temporal Model Evaluation ===")

        temporal_model = joblib.load(temporal_model_path)

        y_pred_temporal = temporal_model.predict(
            X_temporal_test
        )

        y_proba_temporal = temporal_model.predict_proba(
            X_temporal_test
        )[:, 1]

        print(
            "Accuracy:",
            accuracy_score(y_test, y_pred_temporal),
        )

        print(
            "ROC-AUC:",
            roc_auc_score(y_test, y_proba_temporal),
        )

        print(
            classification_report(
                y_test,
                y_pred_temporal,
            )
        )

        # Confusion matrix
        plt.figure(figsize=(6, 5))

        cm_temporal = confusion_matrix(
            y_test,
            y_pred_temporal,
        )

        sns.heatmap(
            cm_temporal,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=["Free", "Congested"],
            yticklabels=["Free", "Congested"],
        )

        plt.title("Temporal Model - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        plt.savefig(
            os.path.join(
                FIGURES_DIR,
                "confusion_matrix_temporal.png",
            ),
            bbox_inches="tight",
        )

        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(
            y_test,
            y_proba_temporal,
        )

        auc_score = roc_auc_score(
            y_test,
            y_proba_temporal,
        )

        plt.figure(figsize=(6, 5))

        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            linewidth=2,
            label=f"AUC = {auc_score:.3f}",
        )

        plt.plot(
            [0, 1],
            [0, 1],
            "k--",
            label="Random Classifier",
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Temporal Model - ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        plt.savefig(
            os.path.join(
                FIGURES_DIR,
                "roc_curve_temporal.png",
            ),
            bbox_inches="tight",
        )

        plt.close()

    else:
        print(f"Temporal model not found: {temporal_model_path}")

    print(f"\nEvaluation completed. Results saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    evaluate_models()