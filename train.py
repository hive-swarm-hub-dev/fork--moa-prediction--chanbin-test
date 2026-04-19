"""
Baseline: predict a small constant for all targets.
Agents should replace this with a better model.
"""
import pandas as pd
import numpy as np

# Load data
train_features = pd.read_csv("data/train_features.csv")
train_targets = pd.read_csv("data/train_targets.csv")
test_features = pd.read_csv("data/test_features.csv")

target_cols = [c for c in train_targets.columns if c != "sig_id"]

# Baseline: predict a small constant for all targets
# (predicting exactly 0.0 causes infinite log loss, so we use a small value)
baseline_pred = 0.001

preds = np.full((len(test_features), len(target_cols)), baseline_pred)
submission = pd.DataFrame(preds, columns=target_cols)
submission.insert(0, "sig_id", test_features["sig_id"].values)

submission.to_csv("submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows x {len(target_cols)} targets")
