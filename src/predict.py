"""Score new loan applications with the current production model.

Example:
    python -m src.predict --age 35 --income 60000 --loan_amount 20000 \
        --credit_score 700 --debt_to_income 0.25 --employment_years 5 --num_credit_lines 4
"""
import argparse

import pandas as pd

from . import config
from .registry import ModelRegistry


def main():
    p = argparse.ArgumentParser(description="Predict default probability")
    for f in config.FEATURES:
        p.add_argument(f"--{f}", type=float, required=True)
    args = p.parse_args()

    reg = ModelRegistry()
    entry = reg.production_entry()
    model = reg.load()
    row = pd.DataFrame([{f: getattr(args, f) for f in config.FEATURES}])
    prob = model.predict_proba(row)[0, 1]
    print(f"Model v{entry['version']} | probability of default = {prob:.1%} -> "
          f"{'REJECT / review' if prob >= 0.5 else 'APPROVE'}")


if __name__ == "__main__":
    main()
