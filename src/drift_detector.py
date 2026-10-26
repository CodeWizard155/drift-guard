"""Drift detection: is today's data different from the training data?

Three checks are combined:

1. PSI  (Population Stability Index) per feature - how much a feature's
   distribution has shifted. Widely used in credit-risk monitoring.
2. KS test (Kolmogorov-Smirnov) per feature - a statistical test of
   whether two samples come from the same distribution.
3. Performance drift - once true labels arrive, has ROC-AUC dropped
   compared with the model's baseline?
"""
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import config


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

    PSI = sum( (a_i - e_i) * ln(a_i / e_i) ) over bins, where e_i and a_i are
    the share of reference and current rows falling in bin i. Bins are the
    deciles of the reference sample.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:          # constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    eps = 1e-6                  # avoid log(0) / division by zero
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_level(value: float) -> str:
    if value >= config.PSI_CRITICAL:
        return "critical"
    if value >= config.PSI_WARNING:
        return "warning"
    return "ok"


@dataclass
class DriftReport:
    week: int
    feature_psi: dict = field(default_factory=dict)
    feature_ks_p: dict = field(default_factory=dict)
    feature_status: dict = field(default_factory=dict)
    drift_share: float = 0.0
    data_drift: bool = False
    auc: float | None = None
    baseline_auc: float | None = None
    performance_drift: bool = False

    @property
    def retrain_needed(self) -> bool:
        return self.data_drift or self.performance_drift

    @property
    def reason(self) -> str:
        reasons = []
        if self.data_drift:
            reasons.append(f"data drift in {self.drift_share:.0%} of features")
        if self.performance_drift:
            reasons.append(f"AUC fell {self.baseline_auc - self.auc:.3f} below baseline")
        return "; ".join(reasons) or "no drift"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["retrain_needed"] = self.retrain_needed
        d["reason"] = self.reason
        return d


class DriftDetector:
    """Compares incoming batches with a stored reference dataset."""

    def __init__(self, reference: pd.DataFrame, features: list[str] | None = None):
        self.features = features or config.FEATURES
        self.reference = reference[self.features].copy()

    def update_reference(self, reference: pd.DataFrame) -> None:
        """Called after retraining: the new training data becomes the reference."""
        self.reference = reference[self.features].copy()

    def check(self, current: pd.DataFrame, week: int,
              auc: float | None = None, baseline_auc: float | None = None) -> DriftReport:
        rep = DriftReport(week=week, auc=auc, baseline_auc=baseline_auc)
        critical = 0
        for f in self.features:
            ref, cur = self.reference[f].values, current[f].values
            value = psi(ref, cur)
            p = ks_2samp(ref, cur).pvalue
            status = psi_level(value)
            # A feature counts as critically drifted only if BOTH tests agree.
            # This cuts false alarms from the very sensitive KS test.
            if status == "critical" and p < config.KS_P_VALUE:
                critical += 1
            elif status == "critical":
                status = "warning"
            rep.feature_psi[f] = round(value, 4)
            rep.feature_ks_p[f] = float(p)
            rep.feature_status[f] = status

        rep.drift_share = critical / len(self.features)
        rep.data_drift = rep.drift_share >= config.DRIFT_SHARE_THRESHOLD
        if auc is not None and baseline_auc is not None:
            rep.performance_drift = (baseline_auc - auc) > config.AUC_DROP_THRESHOLD
        return rep
