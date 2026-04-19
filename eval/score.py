"""Score a submission against hidden test targets using mean column-wise log loss."""
import sys
import numpy as np
import pandas as pd

EPSILON = 1e-15


def column_log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()


def main():
    if len(sys.argv) != 2:
        print("Usage: python score.py <submission.csv>")
        sys.exit(1)

    submission = pd.read_csv(sys.argv[1])
    targets = pd.read_csv("eval/test_targets.csv")

    target_cols = [c for c in targets.columns if c != "sig_id"]

    # Validate submission
    missing_cols = [c for c in target_cols if c not in submission.columns]
    if missing_cols:
        print(f"ERROR: submission missing {len(missing_cols)} columns: {missing_cols[:5]}...")
        print("---")
        print("log_loss:         999.0")
        print(f"correct:          0")
        print(f"total:            {len(target_cols)}")
        return

    if len(submission) != len(targets):
        print(f"ERROR: submission has {len(submission)} rows, expected {len(targets)}")
        print("---")
        print("log_loss:         999.0")
        print(f"correct:          0")
        print(f"total:            {len(target_cols)}")
        return

    # Check for NaN values
    nan_cols = [c for c in target_cols if submission[c].isna().any()]
    if nan_cols:
        print(f"ERROR: submission has NaN values in {len(nan_cols)} columns: {nan_cols[:5]}...")
        print("---")
        print("log_loss:         999.0")
        print(f"correct:          0")
        print(f"total:            {len(target_cols)}")
        return

    # Compute mean column-wise log loss
    losses = []
    for col in target_cols:
        y_true = targets[col].values.astype(float)
        y_pred = submission[col].values.astype(float)
        losses.append(column_log_loss(y_true, y_pred))

    mean_loss = np.mean(losses)

    # Count "good" columns (log loss < 0.01)
    good_cols = sum(1 for l in losses if l < 0.01)

    score = -mean_loss  # negated so higher is better (for Hive leaderboard)

    print(f"Mean column-wise log loss: {mean_loss:.6f}")
    print(f"Best column loss:  {min(losses):.6f}")
    print(f"Worst column loss: {max(losses):.6f}")
    print("---")
    print(f"score:            {score:.6f}")
    print(f"log_loss:         {mean_loss:.6f}")
    print(f"correct:          {good_cols}")
    print(f"total:            {len(target_cols)}")


if __name__ == "__main__":
    main()
