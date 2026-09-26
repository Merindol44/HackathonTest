"""Curriculum training for G1 stair climbing: stairs grow stage by stage.

Starts from the flat-ground 3M-step walker and fine-tunes it through
increasing stair heights (2 cm -> 12 cm). Each stage trains a fixed step
budget, saves a checkpoint, and renders an eval video so progress is
visible in the morning.

Run headless under xvfb-run so the per-stage renders work:
    xvfb-run -a python -u scripts/train_curriculum.py --run-name stairs_curriculum

Restartability: if the run dies mid-curriculum, relaunch with
    --start-stage N   (loads curriculum_stage{N-1} and continues from stage N)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)  # so sibling modules import

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from g1_stairs_env import G1StairsEnv
from train_stairs import KLWatchdogCallback, PPO_KWARGS

# (stair height in meters, PPO timesteps). Total = 4.5M steps.
STAGES = [
    (0.02, 500_000),
    (0.04, 500_000),
    (0.06, 750_000),
    (0.08, 750_000),
    (0.10, 1_000_000),
    (0.12, 1_000_000),
]

BASE_DIR = Path(__file__).resolve().parent.parent
BASE_MODEL = (BASE_DIR / "runs/stairs_overnight/checkpoints"
              / "ppo_stairs_3000000_steps.zip")
BASE_STATS = (BASE_DIR / "runs/stairs_overnight/checkpoints"
              / "ppo_stairs_vecnormalize_3000000_steps.pkl")


def parse_args():
    p = argparse.ArgumentParser(description="Curriculum PPO for G1 stairs")
    p.add_argument("--run-name", type=str, default="stairs_curriculum")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--start-stage", type=int, default=0,
                   help="stage index to (re)start from; loads the previous "
                        "stage's checkpoint when > 0")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = BASE_DIR / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.start_stage == 0:
        model_path = BASE_MODEL
        stats_path = BASE_STATS
    else:
        prev = args.start_stage - 1
        ph = STAGES[prev][0]
        model_path = ckpt_dir / f"curriculum_stage{prev}_h{ph:.2f}.zip"
        stats_path = ckpt_dir / f"curriculum_stage{prev}_vecnormalize.pkl"
    if not Path(model_path).exists():
        sys.exit(f"[curriculum] model not found: {model_path}")
    if not Path(stats_path).exists():
        sys.exit(f"[curriculum] vecnormalize stats not found: {stats_path}")

    model = None
    for idx in range(args.start_stage, len(STAGES)):
        height, timesteps = STAGES[idx]
        print(f"[curriculum] stage {idx}: stair_height={height:.2f} m, "
              f"{timesteps} steps", flush=True)

        venv = make_vec_env(
            lambda h=height: G1StairsEnv(stair_height=h),
            n_envs=args.n_envs, seed=100 + idx, vec_env_cls=SubprocVecEnv)
        # Carry normalization stats across stages (and from the base walker);
        # they keep adapting since training stays enabled.
        vec_env = VecNormalize.load(str(stats_path), venv)

        if model is None:
            print(f"[curriculum] loading base model {model_path}", flush=True)
            model = PPO.load(str(model_path), env=vec_env)
            # PPO.load keeps the checkpoint's timestep counter; fine-tuning
            # continues from there.
        else:
            model.set_env(vec_env)

        ckpt_cb = CheckpointCallback(
            save_freq=max(250_000 // args.n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix=f"ppo_curriculum_s{idx}",
            save_vecnormalize=True,
        )
        kl_watchdog = KLWatchdogCallback(save_path=ckpt_dir,
                                         threshold=2.0, patience=3)
        model.learn(total_timesteps=timesteps,
                    callback=CallbackList([ckpt_cb, kl_watchdog]),
                    reset_num_timesteps=False)

        stage_model = ckpt_dir / f"curriculum_stage{idx}_h{height:.2f}"
        # NOTE: PPO.save() writes the literal path (no auto .zip), while
        # PPO.load() falls back to path+".zip" when the file is missing --
        # so always save WITH the extension to keep save/load symmetric.
        model.save(str(stage_model) + ".zip")
        stats_path = str(ckpt_dir / f"curriculum_stage{idx}_vecnormalize.pkl")
        vec_env.save(stats_path)

        out = run_dir / f"eval_stage{idx}_h{height:.2f}.mp4"
        subprocess.run(
            [sys.executable, str(BASE_DIR / "scripts" / "render_eval.py"),
             "--model", str(stage_model) + ".zip",
             "--vecnormalize", stats_path,
             "--stair-height", str(height),
             "--out", str(out),
             "--episodes", "3", "--fps", "30"],
            check=True,
        )
        vec_env.close()
        print(f"[curriculum] stage {idx} done: {stage_model}.zip + {out.name}",
              flush=True)

    print("[curriculum] ALL STAGES DONE", flush=True)


if __name__ == "__main__":
    main()
