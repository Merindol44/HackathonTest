"""Watch a trained stairs policy LIVE in an interactive MuJoCo window.

Runs on your own PC (Windows / macOS / Linux with a display) — no VM needed:

    git pull
    python scripts/download_g1_model.py   # once: fetches the G1 model (~60 MB)
    pip install mujoco gymnasium stable-baselines3 torch
    python scripts/view_policy.py         # live H14 rollout, 6x0.12m stairs

Controls: left-drag orbits, scroll zooms, ESC quits. Episodes auto-restart.
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Watch a trained stairs policy live")
    p.add_argument("--model", type=str,
                   default="runs/night_h14_full/final_model.zip",
                   help="path to PPO checkpoint (.zip suffix optional)")
    p.add_argument("--vecnormalize", type=str, default=None,
                   help="path to vecnormalize .pkl (default: guessed next to --model)")
    p.add_argument("--n-stairs", type=int, default=6)
    p.add_argument("--step-h", type=float, default=0.12)
    p.add_argument("--harness", type=float, default=0.2)
    p.add_argument("--r-level", type=float, default=5.0)
    p.add_argument("--env", type=str, default="v1", choices=["v1", "v3"],
                   help="reward/env version the checkpoint was trained with")
    return p.parse_args()


def main():
    args = parse_args()
    # Resolve paths against the ORIGINAL cwd: the script chdirs into
    # scripts/ below so the env module imports correctly.
    orig_cwd = Path.cwd()

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (orig_cwd / p)

    os.chdir(Path(__file__).resolve().parent)

    model_path = _resolve(args.model)
    if model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")
    if not model_path.exists():
        sys.exit(f"[view] checkpoint not found: {model_path}\n"
                 "Did you `git pull` the latest branch?")
    vn_path = _resolve(args.vecnormalize) if args.vecnormalize else None
    if vn_path is None:
        cands = list(model_path.parent.glob("*vecnormalize*.pkl"))
        vn_path = cands[0] if cands else None
    if vn_path is None or not vn_path.exists():
        sys.exit(f"[view] VecNormalize stats not found near {model_path}.")

    g1_xml = (Path(__file__).resolve().parent.parent / "assets" /
              "mujoco_menagerie" / "unitree_g1" / "g1.xml")
    if not g1_xml.exists():
        sys.exit(f"[view] G1 model not found at {g1_xml}\n"
                 "Run once:  python scripts/download_g1_model.py")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    if args.env == "v3":
        from g1_stairs_env_v3 import G1StairsEnvV3 as EnvCls
    else:
        from g1_stairs_env_v2 import G1StairsEnvV1 as EnvCls
    import mujoco.viewer

    raw_env = EnvCls(n_stairs=args.n_stairs, step_h=args.step_h,
                     harness=args.harness, r_level=args.r_level)
    venv = VecNormalize.load(str(vn_path), DummyVecEnv([lambda: raw_env]))
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(str(model_path.with_suffix("")), env=venv)
    env = venv.envs[0]
    print(f"[view] {model_path.name}: {args.n_stairs}x{args.step_h}m stairs, "
          f"harness={args.harness} — ESC to quit")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        ep = 0
        while viewer.is_running():
            obs = venv.reset()
            done = False
            ep_rew = 0.0
            while not done and viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, infos = venv.step(action)
                ep_rew += float(reward[0])
                viewer.sync()  # paces to real time
            print(f"[view] episode {ep}: reward={ep_rew:.1f} "
                  f"levels={infos[0].get('levels', '?')}")
            ep += 1


if __name__ == "__main__":
    main()
