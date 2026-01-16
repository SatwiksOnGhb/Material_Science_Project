from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}


def save_metrics(metrics: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))


def plot_pred_vs_actual(y_true, y_pred, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_importance(importances, out_path: Path, title: str, top_k: int = 20) -> None:
    """
    For one-hot encoded datasets, there may be many features.
    We'll plot top_k feature importances by index.
    """
    importances = np.asarray(importances)
    idx = np.argsort(importances)[::-1][:top_k]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.bar(range(len(idx)), importances[idx])
    plt.xlabel("Top features (index)")
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
