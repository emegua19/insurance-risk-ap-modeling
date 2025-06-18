"""
Utilities for saving metrics and generating SHAP summary plots.
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt

def save_metrics(metrics: dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_shap_summary(model, X_sample, feature_names, output_png: str):
    """
    Generate a SHAP summary plot for tree‑based models.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Please install shap to use save_shap_summary()")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure()
    shap.summary_plot(shap_values, X_sample,
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=150)
    plt.close()
