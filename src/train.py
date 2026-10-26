"""Model training and evaluation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def build_model() -> Pipeline:
    """Scaler + gradient-boosted trees, wrapped in one sklearn Pipeline so the
    exact same preprocessing is used in training and in production."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            random_state=config.RANDOM_SEED)),
    ])


def evaluate(model, df: pd.DataFrame) -> dict:
    X, y = df[config.FEATURES], df[config.TARGET]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "default_rate": round(float(np.mean(y)), 4),
        "n": int(len(df)),
    }


def train(df: pd.DataFrame, test_size: float = 0.25):
    """Train on `df` with a hold-out split. Returns (model, holdout_metrics, holdout_df)."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=config.RANDOM_SEED, stratify=df[config.TARGET])
    model = build_model()
    model.fit(train_df[config.FEATURES], train_df[config.TARGET])
    return model, evaluate(model, test_df), test_df
