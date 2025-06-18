import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

def explain_model(model, X, output_dir="outputs/shap", top_n=10):
    """Generate SHAP summary for feature importance."""
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Summary Plot (Bar)
    plt.figure()
    shap.plots.bar(shap_values, max_display=top_n, show=False)
    plt.savefig(os.path.join(output_dir, "shap_bar.png"))
    plt.close()

    # Summary Plot (Beeswarm)
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=top_n, show=False)
    plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"))
    plt.close()

    print(f"[SHAP] Saved SHAP plots to: {output_dir}")
