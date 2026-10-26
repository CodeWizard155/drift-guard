"""Synthetic loan-default data with controllable drift.

We simulate a bank's loan applications arriving week by week.

* Weeks 1-4  : stable  -> same world the model was trained on.
* Weeks 5-8  : DATA drift -> economy changes: incomes rise, credit scores
               fall, people borrow more. Inputs change, the rule linking
               inputs to default stays the same.
* Weeks 9-12 : CONCEPT drift -> the rule itself changes: debt-to-income
               becomes far more important and credit score less important
               (e.g. after an interest-rate shock).

Using synthetic data lets us know exactly when and what kind of drift
happens, so we can check whether the detector finds it.
"""
import numpy as np
import pandas as pd

from . import config


def drift_phase(week: int) -> str:
    """Return the name of the drift phase for a given week (0 = reference)."""
    if week <= 4:
        return "stable"
    if week <= 8:
        return "data_drift"
    return "concept_drift"


def _features(n: int, week: int, rng: np.random.Generator) -> pd.DataFrame:
    # Strength of the data drift grows gradually from week 5 to week 12.
    s = 0.0 if week <= 4 else min((week - 4) / 4, 1.0)

    age = rng.normal(40, 11, n).clip(21, 75)
    income = rng.lognormal(mean=np.log(55_000) + 0.35 * s, sigma=0.45, size=n)
    loan_amount = rng.lognormal(mean=np.log(18_000) + 0.40 * s, sigma=0.6, size=n)
    credit_score = rng.normal(680 - 45 * s, 60, n).clip(300, 850)
    debt_to_income = rng.beta(2 + 1.5 * s, 5, n)
    employment_years = rng.exponential(6, n).clip(0, 40)
    num_credit_lines = rng.poisson(4 + 1.5 * s, n)

    return pd.DataFrame({
        "age": age.round(0),
        "income": income.round(0),
        "loan_amount": loan_amount.round(0),
        "credit_score": credit_score.round(0),
        "debt_to_income": debt_to_income.round(3),
        "employment_years": employment_years.round(1),
        "num_credit_lines": num_credit_lines,
    })


def _default_probability(df: pd.DataFrame, week: int) -> np.ndarray:
    concept = drift_phase(week) == "concept_drift"

    # Before the shock: credit score and loan size drive risk.
    # After the shock: debt-to-income dominates, credit score barely matters,
    # and long employment no longer protects (e.g. layoffs hit senior staff).
    w_credit = -0.025 if not concept else -0.002
    w_dti = 3.0 if not concept else 12.0
    w_loan = 1.4 if not concept else 0.2
    w_emp = -0.10 if not concept else 0.06

    z = (
        -1.4
        + w_credit * (df["credit_score"] - 650)
        + w_dti * (df["debt_to_income"] - 0.3)
        + w_loan * np.log(df["loan_amount"] / df["income"] * 3)
        + w_emp * (df["employment_years"] - 6)
        + 0.08 * (df["num_credit_lines"] - 4)
        - 0.015 * (df["age"] - 40)
    )
    return 1 / (1 + np.exp(-z))


def generate_batch(n: int, week: int, seed: int | None = None) -> pd.DataFrame:
    """Generate one labelled batch of loan applications for a given week."""
    rng = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED + week)
    df = _features(n, week, rng)
    p = _default_probability(df, week)
    df[config.TARGET] = (rng.random(n) < p).astype(int)
    df["week"] = week
    return df


if __name__ == "__main__":
    for w in (0, 6, 10):
        b = generate_batch(1000, w)
        print(f"week {w:2d} ({drift_phase(w):13s}) default rate={b['default'].mean():.2%} "
              f"mean income={b['income'].mean():,.0f}")
