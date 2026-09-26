"""Train a PPO policy for G1 stair climbing (stable-baselines3).

Saves checkpoints, the final model and VecNormalize statistics under
runs/<run-name>/ so evaluation / video rendering can reload everything.

Example:
    python scripts/train_stairs.py --timesteps 3000000 --n-envs 8 \\
        --seed 0 --run-name stairs_overnight
"""

import argparse
import os
import re
import sys
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


class HarnessAnnealCallback(BaseCallback):
    """Smoothly anneal the fall-harness strength inside a single run.

    Linearly interpolates harness from ``h_start`` to ``h_end`` over
    ``anneal_steps`` *new* timesteps of this run (measured from the timestep
    counter at training start, so resumes schedule correctly). Applies the
    value to every env via ``env_method("set_harness", ...)`` at each
    rollout end and logs it as ``harness/cur``.

    Rationale: discrete harness drops between stages (0.2 -> 0.15 -> 0.10)
    shock the policy (KL spike, LR halving, walker regression). A smooth
    ramp lets PPO adapt its balance controller continuously.
    """

    def __init__(self, h_start, h_end, anneal_steps, verbose=0):
        super().__init__(verbose)
        self.h_start = float(h_start)
        self.h_end = float(h_end)
        self.anneal_steps = int(anneal_steps)
        self._t0 = None

    def _on_training_start(self) -> None:
        self._t0 = self.model.num_timesteps
        self._apply(self.h_start)

    def _on_step(self) -> bool:
        return True

    def _apply(self, h):
        try:
            self.model.get_env().env_method("set_harness", h)
        except Exception as exc:  # never kill a run over the schedule
            print(f"[harness-anneal] env_method failed: {exc}", flush=True)
        self.logger.record("harness/cur", h)

    def _on_rollout_end(self) -> bool:
        if self._t0 is None:
            self._t0 = self.model.num_timesteps
        done = self.model.num_timesteps - self._t0
        frac = min(1.0, max(0.0, done / max(1, self.anneal_steps)))
        h = self.h_start + (self.h_end - self.h_start) * frac
        self._apply(h)
        if self.verbose:
            print(f"[harness-anneal] step {done}/{self.anneal_steps} "
                  f"harness={h:.3f}", flush=True)
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
                   help="path to a checkpoint to continue training from; "
                        "the .zip extension is optional and the path is "
                        "resolved against the directory you run from")
    p.add_argument("--env", type=str, default="v0", choices=["v0", "v1", "v3"],
                   help="v0 = original env, v1 = band+clearance rewards with "
                        "parameterized stair geometry, v3 = v1 + all eight "
                        "brainstorm reward ideas (dense progress potential, "
                        "alternating cadence, single-support balance, split "
                        "level bonus, harness budget, foot placement, "
                        "landing softness, pelvis height tracking)")
    p.add_argument("--n-stairs", type=int, default=6,
                   help="v1 only: number of steps (0 = flat ground)")
    p.add_argument("--step-h", type=float, default=0.12,
                   help="v1 only: step height in meters")
    p.add_argument("--track-w", type=float, default=0.0,
                   help="v1 only: reference-gait tracking reward weight")
    p.add_argument("--anti-stand", action="store_true",
                   help="v1 only: truncate episodes with no forward progress")
    p.add_argument("--harness", type=float, default=0.0,
                   help="v1 only: fall-harness support strength 0..1 (training wheels)")
    p.add_argument("--r-level", type=float, default=2.0,
                   help="v1 only: reward per new stair level reached")
    p.add_argument("--harness-end", type=float, default=None,
                   help="v1 only: if set, smoothly anneal harness from "
                        "--harness to this value inside the run")
    p.add_argument("--harness-anneal-steps", type=int, default=0,
                   help="v1 only: new timesteps over which the harness "
                        "anneal ramps (0 = no anneal)")
    return p.parse_args()


def main(orig_cwd):
    args = parse_args()
    run_dir = Path(__file__).resolve().parent.parent / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        if args.env == "v1":
            from g1_stairs_env_v2 import G1StairsEnvV1  # noqa: E402
            return G1StairsEnvV1(n_stairs=args.n_stairs, step_h=args.step_h,
                                 track_w=args.track_w, anti_stand=args.anti_stand,
                                 harness=args.harness, r_level=args.r_level)
        if args.env == "v3":
            from g1_stairs_env_v3 import G1StairsEnvV3  # noqa: E402
            return G1StairsEnvV3(n_stairs=args.n_stairs, step_h=args.step_h,
                                 track_w=args.track_w, anti_stand=args.anti_stand,
                                 harness=args.harness, r_level=args.r_level)
        return G1StairsEnv()

    print(f"[train] env={args.env} n_stairs={args.n_stairs} step_h={args.step_h}")

    venv = make_vec_env(make_env, n_envs=args.n_envs,
                         seed=args.seed, vec_env_cls=SubprocVecEnv)

    # Resolve --resume against the ORIGINAL cwd: main() runs with cwd=scripts/
    # (for the g1_stairs_env import), so a repo-root-relative path like
    # runs/x/checkpoints/ppo_stairs_2000000_steps would otherwise be looked up
    # under scripts/ and silently miss.
    resume_zip = None
    resume_stats = None
    if args.resume:
        rp = Path(args.resume)
        if not rp.is_absolute():
            rp = orig_cwd / rp
        # PPO.load appends .zip itself; accept the path with or without it.
        stem = rp.with_suffix("") if rp.suffix == ".zip" else rp
        if not stem.with_suffix(".zip").exists():
            sys.exit(f"[train] resume checkpoint not found: {stem}.zip")
        resume_zip = str(stem)
        # VecNormalize stats live next to the checkpoint. Try the exact name
        # first, then anything matching; never silently start fresh on a
        # trained policy (fresh normalization = wrongly-scaled inputs =
        # destabilized resume).
        parent = stem.parent
        cands = []
        exact = parent / "vecnormalize.pkl"
        if exact.exists():
            cands.append(exact)

        def _stats_steps(p):
            m = re.search(r"(\d+)_steps", p.name)
            return int(m.group(1)) if m else -1

        # Prefer the stats from the LATEST checkpoint: normalization running
        # averages drift during training, so the newest match the policy best.
        rest = sorted(
            (p for p in parent.glob("*vecnormalize*.pkl") if p != exact),
            key=_stats_steps, reverse=True,
        )
        cands += rest
        if cands:
            resume_stats = str(cands[0])
            print(f"[train] VecNormalize stats: {resume_stats}")
        else:
            print("[train] WARNING: no VecNormalize stats found next to the "
                  "resume checkpoint -- starting with FRESH normalization. "
                  "A trained policy resumed this way sees wrongly-scaled "
                  "inputs and will likely destabilize. Stop now unless that "
                  "is what you want.")

    # On resume, load saved normalization stats onto the RAW env. Wrapping an
    # already-normalized env would normalize twice and corrupt the policy input.
    if resume_stats:
        vec_env = VecNormalize.load(resume_stats, venv)
    else:
        # Normalize observations AND rewards; PPO is sensitive to reward scale
        # and our shaped reward has heterogeneous terms.
        vec_env = VecNormalize(venv, norm_obs=True, norm_reward=True,
                               clip_obs=10.0, gamma=0.99)

    if resume_zip:
        print(f"Resuming from {resume_zip}.zip")
        model = PPO.load(resume_zip, env=vec_env, seed=args.seed)
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
    if args.harness_end is not None and args.harness_anneal_steps > 0:
        anneal_cb = HarnessAnnealCallback(
            h_start=args.harness, h_end=args.harness_end,
            anneal_steps=args.harness_anneal_steps, verbose=1,
        )
        callbacks = CallbackList([checkpoint_cb, kl_watchdog, anneal_cb])
        print(f"[train] harness anneal: {args.harness} -> {args.harness_end} "
              f"over {args.harness_anneal_steps} new timesteps")

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
    _orig_cwd = Path.cwd()
    os.chdir(Path(__file__).resolve().parent)  # so `g1_stairs_env` imports
    main(_orig_cwd)
