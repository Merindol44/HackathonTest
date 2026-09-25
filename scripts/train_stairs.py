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
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
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


class KLWatchdogCallback(BaseCallback):
    """Watch PPO's approx_kl and intervene if it stays high.

    After each rollout, reads the last ``train/approx_kl`` value from the
    PPO logger. If it exceeds ``threshold`` for ``patience`` consecutive
    rollouts it:
      1. prints an alert line,
      2. saves an emergency checkpoint next to the regular checkpoints,
      3. (when halve_lr) multiplies the optimizer learning rate by
         ``lr_factor`` (floored at ``min_lr``) and pins SB3's lr_schedule
         to the new value so the reduction survives future updates.

    Rationale: approx_kl persistently > ~2.0 means PPO updates are too
    aggressive and the run risks collapse; halving the LR is the standard
    rescue. It only affects future updates, never past ones, so a healthy
    run (kl ~1.6) is untouched.
    """

    def __init__(self, save_path, threshold=2.0, patience=3,
                 halve_lr=True, lr_factor=0.5, min_lr=1e-5, verbose=0):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.threshold = threshold
        self.patience = patience
        self.halve_lr = halve_lr
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.bad_rollouts = 0
        self.alerted = False

    def _halve_lr(self):
        """Scale the optimizer LR and pin SB3's schedule. Returns (old, new)."""
        try:
            opt = self.model.policy.optimizer
        except Exception:
            return None
        old = [pg["lr"] for pg in opt.param_groups]
        new = [max(lr * self.lr_factor, self.min_lr) for lr in old]
        for pg, nl in zip(opt.param_groups, new):
            pg["lr"] = nl
        # Pin the schedule so SB3 does not restore the old rate later.
        pinned = new[0]
        try:
            self.model.lr_schedule = lambda _progress_remaining: pinned
        except Exception:
            pass
        return old[0], pinned

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        approx_kl = self.model.logger.name_to_value.get("train/approx_kl")
        if approx_kl is None:
            return True
        if approx_kl > self.threshold:
            self.bad_rollouts += 1
        else:
            self.bad_rollouts = 0
            self.alerted = False
        if self.bad_rollouts >= self.patience and not self.alerted:
            self.alerted = True
            self.save_path.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.save_path / "kl_watchdog_checkpoint"))
            msg = (
                f"[kl-watchdog] approx_kl={approx_kl:.3f} > {self.threshold} "
                f"for {self.patience} consecutive rollouts. "
                f"Emergency checkpoint saved to {self.save_path}."
            )
            if self.halve_lr:
                res = self._halve_lr()
                if res is not None:
                    msg += f" LR {res[0]:.2e} -> {res[1]:.2e}."
            else:
                msg += " Consider LR*0.5 mid-run or a lower-LR restart."
            print(msg, flush=True)
        return True


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

    kl_watchdog = KLWatchdogCallback(
        save_path=ckpt_dir,
        threshold=2.0,
        patience=3,
    )
    callbacks = CallbackList([checkpoint_cb, kl_watchdog])

    print(f"Training {args.timesteps} timesteps x {args.n_envs} envs "
          f"-> {run_dir}")
    # On resume, keep the loaded timestep counter so checkpoints continue at
    # 2.25M/2.5M/... instead of overwriting the earlier run's 250k/500k/... files.
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                reset_num_timesteps=not args.resume)

    model.save(str(run_dir / "final_model"))
    vec_env.save(str(run_dir / "vecnormalize.pkl"))
    print(f"Done. Model + VecNormalize stats saved to {run_dir}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)  # so `g1_stairs_env` imports
    main()
