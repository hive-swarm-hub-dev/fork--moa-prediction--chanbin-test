#!/bin/bash

TIMEOUT=300  # 5 minutes

echo "=== Running train.py ==="
timeout $TIMEOUT python3 train.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "train.py failed or timed out (exit code: $EXIT_CODE)"
    echo "---"
    echo "log_loss:         999.0"
    echo "correct:          0"
    echo "total:            206"
    exit 0
fi

if [ ! -f submission.csv ]; then
    echo "ERROR: submission.csv not found"
    echo "---"
    echo "log_loss:         999.0"
    echo "correct:          0"
    echo "total:            206"
    exit 0
fi

echo "=== Scoring ==="
python3 eval/score.py submission.csv
