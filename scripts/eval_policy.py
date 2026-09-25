"""Multi-seed evaluation of a trained G1 stairs policy (SB3 PPO).

Loads a PPO checkpoint plus its VecNormalize statistics, runs deterministic
episodes across several seeds in G1StairsEnv, and prints the headline numbers
for the results slides: mean success_rate, ep_rew_mean, ep_len_mean.

Example:
    python scripts/eval_policy.py \\
        --model runs/stairs_overnight/checkpoints/ppo_stairs_2000000_steps.zip \\
        --seeds 3 --episodes 10
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed policy evaluation")
    p.add_argument("--model", type=str, required=True,
                   help="path to PPO .zip checkpoint (omit .zip if you like; "
                        "SB3 appends it)")
    p.add_argument("--vecnormalize", type=str, default=None,
                   help="path to vecnormalize .pkl (default: guessed next to --model)")
    p.add_argument("--seeds", type=int, default=3,
                   help="number of env seeds to evaluate (>= 3)")
    p.add_argument("--episodes", type=int, default=10,
                   help="episodes per seed")
    p.add_argument("--out", type=str, default=None,
                   help="optional JSON path for the raw per-seed numbers")
    p.add_argument("--env", type=str, default="v0", choices=["v0", "v1"],
                   help="env version the checkpoint was trained on")
    p.add_argument("--n-stairs", type=int, default=6,
                   help="v1 only: number of steps (0 = flat ground)")
    p.add_argument("--step-h", type=float, default=0.12,
                   help="v1 only: step height in meters")
    return p.parse_args()


def main():
    args = parse_args()
    orig_cwd = Path.cwd()

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (orig_cwd / p)

    # The env module lives next to this script.
    os.chdir(Path(__file__).resolve().parent)

    model_path = _resolve(args.model)
    vn_path = _resolve(args.vecnormalize) if args.vecnormalize else None
    if vn_path is None:
        # CheckpointCallback saves ppo_stairs_vecnormalize_<n>_steps.pkl next to
        # the checkpoint; train_stairs.py also writes vecnormalize.pkl there.
        cands = list(model_path.parent.glob("*vecnormalize*.pkl"))
        vn_path = cands[0] if cands else None
    if vn_path is None or not vn_path.exists():
        sys.exit(f"[eval] VecNormalize stats not found near {model_path}. "
                 "Pass --vecnormalize explicitly.")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from g1_stairs_env import G1StairsEnv

    def make_env():
        if args.env == "v1":
            from g1_stairs_env_v2 import G1StairsEnvV1
            return G1StairsEnvV1(n_stairs=args.n_stairs, step_h=args.step_h)
        return G1StairsEnv()

    per_seed = []
    for seed in range(args.seeds):
        # Fresh stats per seed: VecNormalize.load must wrap a RAW env, never an
        # already-normalized one (the double-wrap bug that killed the 21:40 run).
        venv = DummyVecEnv([lambda: make_env()])
        venv = VecNormalize.load(str(vn_path), venv)
        venv.training = False
        venv.norm_reward = False
        model = PPO.load(str(model_path), env=venv)
        model.set_random_seed(seed)

        # Seed ONCE per seed, BEFORE the episode loop. venv.seed() re-resets
        # the env internally, so calling it after venv.reset() (as before)
        # made every episode after the first start from the identical RNG
        # state, and left seed 0's first episode unseeded entirely.
        venv.seed(seed)
        rewards, lengths, successes = [], [], 0
        for _ in range(args.episodes):
            obs = venv.reset()
            done = False
            ep_rew, ep_len = 0.0, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, infos = venv.step(action)
                ep_rew += float(reward[0])
                ep_len += 1
            rewards.append(ep_rew)
            lengths.append(ep_len)
            successes += int(bool(infos[0].get("is_success", False)))
        venv.close()
        s = {
            "seed": seed,
            "success_rate": successes / args.episodes,
            "ep_rew_mean": float(np.mean(rewards)),
            "ep_len_mean": float(np.mean(lengths)),
            "n_episodes": args.episodes,
        }
        per_seed.append(s)
        print(f"[eval] seed {seed}: success_rate={s['success_rate']:.2f} "
              f"ep_rew_mean={s['ep_rew_mean']:.1f} "
              f"ep_len_mean={s['ep_len_mean']:.0f}")

    success_rates = np.array([s["success_rate"] for s in per_seed])
    print("----")
    print(f"[eval] seeds={args.seeds} episodes/seed={args.episodes}")
    print(f"[eval] success_rate  mean={success_rates.mean():.3f}")
    print(f"[eval] ep_rew_mean   mean={np.mean([s['ep_rew_mean'] for s in per_seed]):.1f}")
    print(f"[eval] ep_len_mean   mean={np.mean([s['ep_len_mean'] for s in per_seed]):.0f}")
    print(f"[eval] SUCCESS_RATE (deck number): "
          f"{success_rates.mean():.3f}  ({successes_sum(per_seed)}/{args.seeds * args.episodes} episodes)")

    if args.out:
        import json
        out = _resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(per_seed, indent=2))
        print(f"[eval] per-seed numbers -> {out}")


def successes_sum(per_seed):
    return int(round(sum(s["success_rate"] * s["n_episodes"] for s in per_seed)))


if __name__ == "__main__":
    main()
