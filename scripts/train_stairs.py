"""Train a PPO policy for G1 stair climbing (stable-baselines3).

Saves checkpoints, the final model and VecNormalize statistics under
runs/<run-name>/ so evaluation / video rendering can reload everything.

Example:
    python scripts/train_stairs.py --timesteps 3000000 --n-envs 8 \\
        --seed 0 --run-name stairs_overnight
"""

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from g1_stairs_env import G1StairsEnv  # noqa: E402  (run from scripts/)

# Hyperparameters chosen for a 29-DoF humanoid trained on CPU:
# - n_steps=2048 with several parallel envs -> large, low-variance batches
# - batch 64 / 10 epochs: standard PPO, fits CPU training
# - lr 3e-4, clip 0.2, gamma 0.99, gae 0.95: SB3 defaults, robust baseline
# - [256, 256] MLP: enough capacity without slowing CPU updates too much
PPO_KWARGS = dict(
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    n_epochs=10,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
)


def parse_args():
    p = argparse.ArgumentParser(description="PPO training for G1 stair climbing")
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default="stairs_ppo")
    p.add_argument("--checkpoint-freq", type=int, default=250_000,
                   help="timesteps between checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="path to a .zip model to continue training from")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(__file__).resolve().parent.parent / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        return G1StairsEnv()

    vec_env = make_vec_env(make_env, n_envs=args.n_envs,
                           seed=args.seed, vec_env_cls=SubprocVecEnv)
    # Normalize observations AND rewards; PPO is sensitive to reward scale and
    # our shaped reward has heterogeneous terms.
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, gamma=0.99)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec_env, seed=args.seed)
        # VecNormalize stats are stored next to the resumed model if present.
        stats = Path(args.resume).parent / "vecnormalize.pkl"
        if stats.exists():
            vec_env = VecNormalize.load(str(stats), vec_env)
            model.set_env(vec_env)
    else:
        model = PPO("MlpPolicy", vec_env, seed=args.seed, **PPO_KWARGS)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix="ppo_stairs",
        save_vecnormalize=True,
    )

    print(f"Training {args.timesteps} timesteps x {args.n_envs} envs "
          f"-> {run_dir}")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)

    model.save(str(run_dir / "final_model"))
    vec_env.save(str(run_dir / "vecnormalize.pkl"))
    print(f"Done. Model + VecNormalize stats saved to {run_dir}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)  # so `g1_stairs_env` imports
    main()
