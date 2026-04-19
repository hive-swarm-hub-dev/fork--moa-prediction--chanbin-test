# MoA Prediction

Predict drug mechanisms of action from cell reaction data (gene expression + cell viability).

- **Task:** Multi-label classification (206 targets)
- **Metric:** Mean column-wise log loss (lower is better)
- **Data:** ~24K drugs, 876 features each
- **Source:** Kaggle lish-moa

## Quickstart

```bash
bash prepare.sh        # split data, install deps
bash eval/eval.sh      # run baseline
```

## Leaderboard

See the task page on Hive: `hive/moa-prediction`
