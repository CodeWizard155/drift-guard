"""DriftGuard: automated monitoring + retraining pipeline.

Run:  python -m src.pipeline

Flow for every production week
    1. New batch of loan applications arrives (with labels at week end).
    2. Production model scores the batch -> ROC-AUC, accuracy, F1.
    3. Drift detector compares the batch with the reference (training) data.
    4. If data drift OR performance drift -> alert + automatic retraining
       on the most recent weeks of data.
    5. Candidate model is validated against the production model on the
       same hold-out data; it is promoted only if it is better.
    6. Everything is logged (CSV/JSON) and registered (model registry).

A "static" model that is never retrained runs alongside as a baseline, so we
can measure how much the MLOps loop actually helps.
"""
import json
import shutil
import time
from datetime import datetime

import pandas as pd

from . import config
from .data_generator import drift_phase, generate_batch
from .drift_detector import DriftDetector
from .registry import ModelRegistry
from .train import evaluate, train


def log(msg: str, alerts=None) -> None:
    print(msg, flush=True)
    if alerts is not None:
        alerts.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")


def run(reset: bool = True) -> dict:
    start = time.time()
    if reset:
        shutil.rmtree(config.MODEL_DIR, ignore_errors=True)
    for d in (config.DATA_DIR, config.MODEL_DIR, config.REPORT_DIR, config.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry()
    alerts = open(config.REPORT_DIR / "alerts.log", "w")

    # ---- Week 0: train the first model on reference data -----------------
    reference = generate_batch(config.REFERENCE_SIZE, week=0)
    reference.to_csv(config.DATA_DIR / "reference.csv", index=False)
    model, metrics, _ = train(reference)
    v = registry.register(model, metrics, trained_on="week 0 (reference)", note="initial model")
    registry.promote(v)
    static_model = model                       # never retrained
    baseline_auc = metrics["auc"]
    detector = DriftDetector(reference)

    log("=" * 78)
    log(" DriftGuard - Model Drift Detection & Automated Retraining Pipeline")
    log("=" * 78)
    log(f"[week 0] trained model v{v} on {len(reference)} rows | hold-out AUC={metrics['auc']:.3f} "
        f"acc={metrics['accuracy']:.3f} F1={metrics['f1']:.3f} -> PRODUCTION")

    rows, reports, history = [], [], []
    for week in range(1, config.NUM_WEEKS + 1):
        batch = generate_batch(config.BATCH_SIZE, week)
        batch.to_csv(config.DATA_DIR / f"batch_week_{week:02d}.csv", index=False)
        history.append(batch)

        prod_entry = registry.production_entry()
        prod_model = registry.load()
        m_prod = evaluate(prod_model, batch)
        m_static = evaluate(static_model, batch)

        rep = detector.check(batch, week, auc=m_prod["auc"], baseline_auc=baseline_auc)
        drifted = [f for f, s in rep.feature_status.items() if s == "critical"]

        log("-" * 78)
        log(f"[week {week:2d}] phase={drift_phase(week):13s} model=v{prod_entry['version']} "
            f"AUC={m_prod['auc']:.3f} (static {m_static['auc']:.3f}) "
            f"drifted features={len(drifted)}/{len(config.FEATURES)}")
        if drifted:
            log("          critical PSI: " + ", ".join(
                f"{f}={rep.feature_psi[f]:.2f}" for f in drifted))

        action, new_version = "none", None
        if rep.retrain_needed:
            log(f"  ALERT  week {week}: {rep.reason} -> triggering retraining", alerts)
            window = pd.concat(history[-config.RETRAIN_WINDOW_WEEKS:], ignore_index=True)
            cand, cand_metrics, holdout = train(window)
            prod_on_holdout = evaluate(prod_model, holdout)
            new_version = registry.register(
                cand, cand_metrics,
                trained_on=f"weeks {max(1, week - config.RETRAIN_WINDOW_WEEKS + 1)}-{week}",
                note=rep.reason)
            if cand_metrics["auc"] >= prod_on_holdout["auc"] + config.MIN_IMPROVEMENT:
                registry.promote(new_version)
                detector.update_reference(window)
                baseline_auc = cand_metrics["auc"]
                action = "retrained_promoted"
                log(f"  RETRAIN candidate v{new_version} AUC={cand_metrics['auc']:.3f} vs "
                    f"production {prod_on_holdout['auc']:.3f} on hold-out -> PROMOTED", alerts)
            else:
                registry.reject(new_version)
                action = "retrained_rejected"
                log(f"  RETRAIN candidate v{new_version} AUC={cand_metrics['auc']:.3f} not better "
                    f"than {prod_on_holdout['auc']:.3f} -> REJECTED", alerts)
        else:
            log(f"          status: OK ({rep.reason})")

        rows.append({
            "week": week,
            "phase": drift_phase(week),
            "model_version": prod_entry["version"],
            "auc_mlops": m_prod["auc"],
            "auc_static": m_static["auc"],
            "accuracy_mlops": m_prod["accuracy"],
            "accuracy_static": m_static["accuracy"],
            "f1_mlops": m_prod["f1"],
            "f1_static": m_static["f1"],
            "default_rate": m_prod["default_rate"],
            "drift_share": round(rep.drift_share, 3),
            "data_drift": rep.data_drift,
            "performance_drift": rep.performance_drift,
            "action": action,
            "new_version": new_version,
            **{f"psi_{f}": rep.feature_psi[f] for f in config.FEATURES},
        })
        reports.append(rep.to_dict())

    alerts.close()
    df = pd.DataFrame(rows)
    df.to_csv(config.REPORT_DIR / "monitoring_log.csv", index=False)
    (config.REPORT_DIR / "drift_reports.json").write_text(json.dumps(reports, indent=2))

    summary = {
        "weeks": config.NUM_WEEKS,
        "mean_auc_static": round(df["auc_static"].mean(), 4),
        "mean_auc_mlops": round(df["auc_mlops"].mean(), 4),
        "min_auc_static": round(df["auc_static"].min(), 4),
        "min_auc_mlops": round(df["auc_mlops"].min(), 4),
        "final_auc_static": float(df["auc_static"].iloc[-1]),
        "final_auc_mlops": float(df["auc_mlops"].iloc[-1]),
        "mean_f1_static": round(df["f1_static"].mean(), 4),
        "mean_f1_mlops": round(df["f1_mlops"].mean(), 4),
        "alerts": int((df["action"] != "none").sum()),
        "models_promoted": int((df["action"] == "retrained_promoted").sum()),
        "models_rejected": int((df["action"] == "retrained_rejected").sum()),
        "false_alarms_in_stable_weeks": int(((df["phase"] == "stable") & (df["action"] != "none")).sum()),
        "first_alert_week": int(df.loc[df["action"] != "none", "week"].min()) if (df["action"] != "none").any() else None,
        "runtime_seconds": round(time.time() - start, 1),
        "registry": registry.entries,
    }
    (config.REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    log("=" * 78)
    log(" SUMMARY")
    log(f"  Mean AUC   static model: {summary['mean_auc_static']:.3f}   "
        f"DriftGuard (MLOps): {summary['mean_auc_mlops']:.3f}")
    log(f"  Final AUC  static model: {summary['final_auc_static']:.3f}   "
        f"DriftGuard (MLOps): {summary['final_auc_mlops']:.3f}")
    log(f"  Alerts: {summary['alerts']} | promoted: {summary['models_promoted']} | "
        f"rejected: {summary['models_rejected']} | false alarms (stable weeks): "
        f"{summary['false_alarms_in_stable_weeks']}")
    log(f"  Production model: v{registry.production_entry()['version']} | "
        f"runtime {summary['runtime_seconds']} s")
    log("=" * 78)
    return summary


if __name__ == "__main__":
    run()
