# MoA Prediction

Predict drug mechanisms of action from cell reaction data. Minimize mean column-wise log loss.

## Setup

1. **Read the in-scope files**:
   - `train.py` — training and prediction script. You modify this.
   - `eval/eval.sh` — runs evaluation. Do not modify.
   - `prepare.sh` — installs deps and splits data. Do not modify.
2. **Run prepare**: `bash prepare.sh` to install packages and prepare data.
3. **Verify data exists**: Check that `data/` contains `train_features.csv`, `train_targets.csv`, and `test_features.csv`.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
5. **Run baseline**: `bash eval/eval.sh` to establish the starting score.

## The benchmark

Given 876 measurements of how cells react to a drug (772 gene expressions + 100 cell viabilities + controls), predict which of 206 biological mechanisms the drug triggers. Each drug can trigger multiple mechanisms (multi-label). Targets are extremely sparse — 99.7% are zeros. Training set has ~19K rows, test set has ~4.8K rows.

## Data

- `data/train_features.csv` — features per drug (sig_id, cp_type, cp_time, cp_dose, g-0..g-771, c-0..c-99)
- `data/train_targets.csv` — 206 binary targets per drug (sig_id, then named columns like `5-alpha_reductase_inhibitor`, `acetylcholine_receptor_agonist`, etc.)
- `data/test_features.csv` — same features for held-out drugs (no targets provided)
- `cp_type`: "trt_cp" (treatment) or "ctl_vehicle" (control). Controls have no MoA — all targets should be 0.

## Experimentation

**What you CAN do:**
- Modify `train.py` — use any model, feature engineering, or strategy you want.
- Use any packages in `requirements.txt` (numpy, pandas, scikit-learn, lightgbm).
- Add new packages to `requirements.txt` if needed.

**What you CANNOT do:**
- Modify `eval/`, `prepare.sh`, or data files.
- **NEVER use `eval/test_targets.csv` for training, feature engineering, or any purpose other than running `eval/eval.sh`.** This file exists only for local evaluation. Your model must learn to generalize, not memorize the test set.
- Use internet access during eval.

**The goal: maximize score.** Score is negated mean column-wise log loss (-log_loss). Higher is better. A score of -0.02 is better than -0.03.

**Simplicity criterion**: All else being equal, simpler is better.

## Hints

- Control samples (`cp_type == "ctl_vehicle"`) always have all-zero targets. Handle them separately.
- PCA on gene expression features often helps.
- Consider multi-output models (one model for all 206 targets) vs. per-target models.
- Target correlations exist — some mechanisms co-occur.

## Output format

`train.py` must produce `submission.csv` with columns: sig_id, then all 206 target columns with predicted probabilities (0 to 1).

Eval output:
```
---
score:            <value>    (negated log_loss, higher is better — use this for hive run submit)
log_loss:         <value>    (raw log loss, lower is better)
correct:          <N>
total:            206
```
