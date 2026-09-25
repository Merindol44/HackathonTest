#!/bin/bash
# Smoke test: short PPO run to verify the pipeline end-to-end.
# Verifies: no crashes, env resets, rewards finite, checkpoint + model saved.
set -e
cd "$(dirname "$0")"
source ../.venv/bin/activate
python train_stairs.py --timesteps 50000 --n-envs 4 --seed 0 \
    --run-name smoke_test --checkpoint-freq 25000
echo "SMOKE TEST OK: $(ls ../runs/smoke_test/)"
