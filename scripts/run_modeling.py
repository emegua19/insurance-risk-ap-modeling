#!/usr/bin/env python3
"""
End‑to‑end Task‑4 modelling pipeline driven by YAML config.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import pandas as pd
import logging
import os
import sys
import warnings

# ------------------------------------------------------------------ #
# Local imports
# ------------------------------------------------------------------ #
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from src.hypothesis_testing.metrics import add_metrics
from src.modeling.features import FeatureBuilder
from src.modeling.models import ClassifierModel, RegressorModel
from src.modeling.evaluation import save_metrics, save_shap_summary


# ------------------------------------------------------------------ #
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger()


# ------------------------------------------------------------------ #
def main(cfg_path: str) -> None:
    cfg = load_yaml(cfg_path)
    log = setup_logger()

    # ---------------- 1. Load data & add metrics ------------------ #
    df = pd.read_csv(cfg["data"]["cleaned_path"], low_memory=False)
    df = add_metrics(df)

    # Log portfolio‑level KPIs
    log.info("Average Margin: %.2f", df["margin"].mean())
    log.info("Average Loss Ratio: %.4f", df["loss_ratio"].mean())
    log.info("Claim Frequency: %.4f", df["ClaimOccurred"].mean())

    # ---------------- 2. Ensure output dirs ---------------------- #
    Path(cfg["output"]["model_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["reports_dir"]).mkdir(parents=True, exist_ok=True)

    # ---------------- 3. Feature lists --------------------------- #
    num_cols = df.select_dtypes("number").columns.tolist()
    drops = ["ClaimOccurred", "TotalPremium", "TotalClaims",
             "margin", "loss_ratio", "claim_frequency"]
    for d in drops:
        num_cols = [c for c in num_cols if c != d]

    # Force categoricals to string to avoid mixed‑type OneHotEncoder error
    cat_cols = df.select_dtypes(["object", "category"]).columns.tolist()
    df[cat_cols] = df[cat_cols].astype(str)

    # ---------------- 4. Build pre‑processor --------------------- #
    fb = FeatureBuilder(
        numeric_cols=num_cols,
        cat_cols=cat_cols,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify_col=cfg["split"]["stratify"]
    )
    preproc = fb.build_preprocessor()

    # Silence StandardScaler divide‑by‑zero warnings
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    # ---------------- 5. CLASSIFIER ------------------------------ #
    X_tr, X_te, y_tr, y_te = fb.train_test_split(df, target="ClaimOccurred")
    clf_conf = cfg["classifier"]
    clf_w = ClassifierModel(clf_conf["model"], clf_conf["params"])

    from sklearn.pipeline import Pipeline
    clf_pipe = Pipeline([("prep", preproc), ("model", clf_w.model)])
    clf_pipe.fit(X_tr, y_tr)

    y_pred_proba = clf_pipe.predict_proba(X_te)[:, 1]
    clf_metrics = clf_w.evaluate(y_pred_proba, y_te)

    clf_w.save(Path(cfg["output"]["model_dir"]) / "classifier.joblib")
    save_metrics(clf_metrics,
                 Path(cfg["output"]["reports_dir"]) / "classifier_metrics.json")
    log.info("Classifier metrics: %s", clf_metrics)

    # SHAP (optional)
    if cfg["output"].get("enable_shap", False):
        try:
            sample = preproc.transform(
                X_te.sample(cfg["output"]["shap_sample_size"], random_state=0))
            save_shap_summary(
                clf_w.model, sample, preproc.get_feature_names_out(),
                str(Path(cfg["output"]["reports_dir"]) / "shap_classifier.png"))
            log.info("SHAP summary for classifier saved")
        except Exception as exc:
            log.warning("SHAP classifier failed: %s", exc)

    # ---------------- 6. PREMIUM REGRESSOR ----------------------- #
    X_tr_r, X_te_r, y_tr_r, y_te_r = fb.train_test_split(df, target="TotalPremium")
    reg_conf = cfg["regressor"]
    reg_w = RegressorModel(reg_conf["model"], reg_conf["params"])
    reg_pipe = Pipeline([("prep", preproc), ("model", reg_w.model)])
    reg_pipe.fit(X_tr_r, y_tr_r)

    reg_predictions = reg_pipe.predict(X_te_r)
    reg_metrics = reg_w.evaluate(reg_predictions, y_te_r)
    reg_w.save(Path(cfg["output"]["model_dir"]) / "regressor.joblib")
    save_metrics(reg_metrics,
                 Path(cfg["output"]["reports_dir"]) / "regressor_metrics.json")
    log.info("Regressor metrics: %s", reg_metrics)

    if cfg["output"].get("enable_shap", False):
        try:
            sample_r = preproc.transform(
                X_te_r.sample(cfg["output"]["shap_sample_size"], random_state=0))
            save_shap_summary(
                reg_w.model, sample_r, preproc.get_feature_names_out(),
                str(Path(cfg["output"]["reports_dir"]) / "shap_regressor.png"))
            log.info("SHAP summary for regressor saved")
        except Exception as exc:
            log.warning("SHAP regressor failed: %s", exc)

    # ---------------- 7. SEVERITY REGRESSOR ---------------------- #
    sev_df = df[df["TotalClaims"] > 0].copy()
    if not sev_df.empty:
        X_tr_s, X_te_s, y_tr_s, y_te_s = fb.train_test_split(
            sev_df, target="TotalClaims")
        sev_conf = cfg["severity"]
        sev_w = RegressorModel(sev_conf["model"], sev_conf["params"])
        sev_pipe = Pipeline([("prep", preproc), ("model", sev_w.model)])
        sev_pipe.fit(X_tr_s, y_tr_s)

        sev_pred = sev_pipe.predict(X_te_s)
        sev_metrics = sev_w.evaluate(sev_pred, y_te_s)
        sev_w.save(Path(cfg["output"]["model_dir"]) / "severity_regressor.joblib")
        save_metrics(sev_metrics,
                     Path(cfg["output"]["reports_dir"]) / "severity_metrics.json")
        log.info("Severity metrics: %s", sev_metrics)

        if cfg["output"].get("enable_shap", False):
            try:
                sample_s = preproc.transform(
                    X_te_s.sample(cfg["output"]["shap_sample_size"], random_state=0))
                save_shap_summary(
                    sev_w.model, sample_s, preproc.get_feature_names_out(),
                    str(Path(cfg["output"]["reports_dir"]) / "shap_severity.png"))
                log.info("SHAP summary for severity saved")
            except Exception as exc:
                log.warning("SHAP severity failed: %s", exc)
    else:
        sev_metrics = {}
        log.warning("No positive claims → severity model skipped")

    # ---------------- 8. Markdown Report ------------------------ #
    report_md = Path(cfg["output"]["reports_dir"]) / "model_comparison.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Model Comparison\n\n")
        f.write("| Model | Metric | Score |\n|---|---|---|\n")
        rows = [
            ("Classifier ROC‑AUC", clf_metrics["roc_auc"]),
            ("Classifier F1", clf_metrics["f1"]),
            ("Premium RMSE", reg_metrics["rmse"]),
            ("Severity RMSE", sev_metrics.get("rmse", "n/a")),
        ]
        for name, score in rows:
            f.write(f"| {name} | &nbsp; | {score:.4f} |\n")
    log.info("Markdown report written to %s", report_md)
    log.info("✅ Modeling pipeline finished")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/modeling_config.yaml")
    main(parser.parse_args().config)
