# SKF Hackathon — Training Industrial Humanoids: Stair Climbing

Gbg Tech Week x Chalmers Hackathon 2026 · SKF challenge:
*Analyze and compare learning methods suitable for industrial humanoids.*

This repo contains a complete reinforcement-learning pipeline that trains a
Unitree G1 humanoid to climb stairs in MuJoCo simulation, plus a written
comparison of learning methods for industrial humanoids.

## Quick start

```bash
# 1. Get the G1 robot model (~60 MB, kept out of the repo)
bash scripts/download_g1_model.sh

# 2. Install dependencies (Python 3.10+)
pip install mujoco gymnasium stable-baselines3 torch

# 3. Smoke test (~50k steps, a few minutes)
bash scripts/smoke_test.sh

# 4. Full training run
python scripts/train_stairs.py --timesteps 3000000 --n-envs 8 --run-name stairs_overnight

# 5. Render a video of a trained checkpoint
python scripts/render_eval.py --model runs/<run-name>/final_model.zip --out eval.mp4
```

## Layout

| Path | What it is |
|---|---|
| `scripts/g1_stairs_env.py` | Gymnasium environment: G1 + 6-step staircase, reward shaping |
| `scripts/train_stairs.py` | PPO training script (stable-baselines3) |
| `scripts/render_eval.py` | Headless MP4 rendering of a trained policy |
| `scripts/smoke_test.sh` | Short end-to-end pipeline test |
| `scripts/download_g1_model.sh` | Fetches the G1 MJCF model (MuJoCo Menagerie) |
| `scripts/verify_g1_loads.py` | Sanity check: model loads and steps |
| `docs/learning_methods_comparison.md` | Analysis: 7 learning methods vs industrial requirements |
| `runs/` | Training outputs (checkpoints, logs; gitignored) |

## Approach

Reinforcement learning (PPO) in MuJoCo, sim-to-real oriented:
proprioceptive observations only, position-servo actions around a standing
keyframe, dense progress reward with fall termination. See
`docs/learning_methods_comparison.md` for why RL + sim-to-real was chosen and
how the alternatives compare.
