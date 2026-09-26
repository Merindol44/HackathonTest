"""Deterministic probe of a trained G1 policy on stairs.

Loads a PPO checkpoint (+VecNormalize), runs N deterministic episodes with a
given harness strength, and records per-episode: stair levels climbed,
max forward distance (pelvis_x), total reward, episode length.
Writes a JSON report and prints a summary line.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


def make_env(args):
    from g1_stairs_env_v2 import G1StairsEnvV1
    return G1StairsEnvV1(n_stairs=args.n_stairs, step_h=args.step_h,
                        anti_stand=args.anti_stand, harness=args.harness)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--vecnormalize", default=None)
    p.add_argument("--n-stairs", type=int, default=3)
    p.add_argument("--step-h", type=float, default=0.06)
    p.add_argument("--anti-stand", action="store_true")
    p.add_argument("--harness", type=float, default=0.0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    venv = DummyVecEnv([lambda: make_env(args)])
    vn_path = args.vecnormalize or str(Path(args.model).with_suffix("") ).replace(".zip", "")  # fallback
    vn_candidates = [args.vecnormalize,
                     Path(str(args.model).replace(".zip", "")).parent / "vecnormalize.pkl"]
    for c in vn_candidates:
        if c and Path(c).exists():
            venv = VecNormalize.load(str(c), venv)
            venv.training = False
            venv.norm_reward = False
            break

    model = PPO.load(args.model)
    # Seed ONCE before the episode loop (matches eval_policy.py's proven pattern).
    venv.seed(args.seed)
    eps = []
    for i in range(args.episodes):
        obs = venv.reset()
        done = False
        ep_rew, ep_len, max_x, levels = 0.0, 0, -1e9, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            ep_rew += float(reward[0])
            ep_len += 1
            inf = info[0]
            levels = max(levels, int(inf.get("levels", 0)))
            max_x = max(max_x, float(inf.get("pelvis_x", 0.0)))
        eps.append({"levels": levels, "max_x": round(max_x, 3),
                    "reward": round(ep_rew, 1), "length": ep_len})

    levels = np.array([e["levels"] for e in eps])
    rep = {"harness": args.harness, "model": args.model,
           "n": len(eps),
           "max_levels": int(levels.max()),
           "mean_levels": round(float(levels.mean()), 2),
           "frac_ge1": round(float((levels >= 1).mean()), 2),
           "mean_max_x": round(float(np.mean([e["max_x"] for e in eps])), 2),
           "mean_reward": round(float(np.mean([e["reward"] for e in eps])), 1),
           "mean_len": round(float(np.mean([e["length"] for e in eps])), 0),
           "episodes": eps}
    print(f"[probe] harness={args.harness} max_levels={rep['max_levels']} "
          f"mean_levels={rep['mean_levels']} frac>=1={rep['frac_ge1']} "
          f"mean_max_x={rep['mean_max_x']}m mean_rew={rep['mean_reward']} "
          f"mean_len={rep['mean_len']}")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"[probe] -> {args.out}")


if __name__ == "__main__":
    main()
