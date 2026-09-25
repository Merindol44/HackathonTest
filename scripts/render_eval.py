"""Render an evaluation rollout of a trained stairs policy to MP4.

Loads a PPO checkpoint plus its VecNormalize statistics, runs deterministic
rollouts in G1StairsEnv, and writes frames to an MP4 file.

Headless rendering: tries MUJOCO_GL=osmesa, then egl. On machines with no GL
at all (e.g. our headless training box), rendering is skipped with a clear
message -- run this script on a laptop with a display instead, where
mujoco.viewer / the default GL backend works out of the box.

Example:
    python scripts/render_eval.py --model runs/smoke_test/checkpoints/ppo_stairs_25000_steps.zip \\
        --out runs/smoke_test/eval.mp4 --episodes 2
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _try_init_renderer(model):
    """Return a mujoco.Renderer, or None if no GL backend is available."""
    import mujoco
    last_err = None
    for backend in ("osmesa", "egl"):
        os.environ["MUJOCO_GL"] = backend
        try:
            r = mujoco.Renderer(model, height=480, width=640)
            print(f"[render] using MUJOCO_GL={backend}")
            return r
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[render] MUJOCO_GL={backend} failed: {type(e).__name__}")
    print(f"[render] no headless GL backend available ({last_err}). "
          "Run on a machine with a display instead.")
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Render policy rollout to MP4")
    p.add_argument("--model", type=str, required=True,
                   help="path to PPO .zip checkpoint")
    p.add_argument("--vecnormalize", type=str, default=None,
                   help="path to vecnormalize .pkl (default: guessed next to --model)")
    p.add_argument("--out", type=str, default="eval.mp4")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(Path(__file__).resolve().parent)

    model_path = Path(args.model)
    vn_path = Path(args.vecnormalize) if args.vecnormalize else None
    if vn_path is None:
        # CheckpointCallback saves ppo_stairs_vecnormalize_<n>_steps.pkl
        # next to the checkpoint; train_stairs.py saves vecnormalize.pkl.
        cands = list(model_path.parent.glob("*vecnormalize*.pkl"))
        vn_path = cands[0] if cands else None
    if vn_path is None or not vn_path.exists():
        sys.exit(f"[render] VecNormalize stats not found near {model_path}. "
                 "Pass --vecnormalize explicitly.")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from g1_stairs_env import G1StairsEnv

    venv = DummyVecEnv([lambda: G1StairsEnv()])
    venv = VecNormalize.load(str(vn_path), venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(str(model_path), env=venv)

    renderer = _try_init_renderer(venv.envs[0].model)
    if renderer is None:
        sys.exit(1)

    import imageio
    frames = []
    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = venv.step(action)
            ep_rew += float(reward[0])
            renderer.update_scene(venv.envs[0].data)
            frames.append(renderer.render())
        success = bool(infos[0].get("is_success", False))
        print(f"[render] episode {ep}: reward={ep_rew:.1f} success={success} "
              f"frames={len(frames)}")
    renderer.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Env runs at 100 Hz; downsample to target fps.
    stride = max(100 // args.fps, 1)
    imageio.mimsave(str(out), frames[::stride], fps=args.fps,
                    codec="libx264", quality=8)
    print(f"[render] wrote {out} ({len(frames[::stride])} frames)")


if __name__ == "__main__":
    main()
