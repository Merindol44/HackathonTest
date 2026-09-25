# Code review — `muse/review-improvements` branch (2026-09-25)

Static review of the HackathonTest repo (no runtime testing: this machine has
no mujoco/gymnasium/SB3 and no G1 model). All scripts still compile
(`python -m py_compile`); no training logic was changed — training strategy
decisions stay with the team.

## What the repo is for

SKF hackathon (Gbg Tech Week x Chalmers 2026): *analyze and compare learning
methods suitable for industrial humanoids*. Two deliverables:

1. **A working RL pipeline** — Unitree G1 learns stair climbing in MuJoCo via
   PPO (stable-baselines3): `g1_stairs_env.py` (G1 + 6-step staircase,
   dense reward: forward velocity + upward velocity − energy − tilt, fall
   termination) → `train_stairs.py` (8 envs, KL watchdog) → `eval_policy.py`
   (multi-seed success rate) → `render_eval.py` (MP4, tracking camera).
2. **A written analysis** — `docs/learning_methods_comparison.md` compares 7
   methods; `docs/post_mortem.md` documents the overnight run honestly
   (reward −53 → +27, success_rate 0: the dense-reward "locomotion trap").

## What's solid

- Env code is clean and well-documented: action mapping around the standing
  keyframe, finite-reward guard, MjSpec composition so mesh paths resolve.
- The `--resume` machinery (timestep-counter fix, double-wrap guard, KL
  watchdog with LR halving) directly encodes lessons from the two dead runs.
- Eval/render scripts share the same VecNormalize conventions and already
  guard the double-wrap bug.

## Issues found and fixed on this branch

1. **`train_stairs.py` — `--resume` could silently train on fresh
   normalization.** Three compounding problems: (a) the path was resolved
   relative to `scripts/` (the script chdirs there for the env import), so a
   repo-root-relative `--resume runs/...` missed and fell back to *fresh*
   VecNormalize stats with no error; (b) stats lookup only tried the exact
   name `vecnormalize.pkl`, never the `ppo_stairs_vecnormalize_*_steps.pkl`
   files CheckpointCallback actually writes; (c) passing `--resume` *with*
   the `.zip` extension broke `PPO.load` (this burned the team twice —
   see gotcha #1 in `tony.txt`). Fixed: resolve `--resume` against the
   original cwd, accept the path with or without `.zip`, try the exact stats
   name then glob `*vecnormalize*.pkl` next to the checkpoint, fail loudly
   if the checkpoint is missing, and print a clear WARNING (instead of
   silence) when no stats are found.
2. **`eval_policy.py` — per-seed episodes were not independent.**
   `venv.seed(seed)` ran *after* `venv.reset()`, and since `venv.seed()`
   re-resets the env internally, every episode after the first in each seed
   loop started from the identical RNG state (seed 0's first episode was
   unseeded entirely). Seed labels were misleading and episode counts were
   inflated. Fixed: seed once per seed, before the episode loop.
3. **`verify_g1_loads.py` — stale docstring.** The hardcoded machine path
   was fixed in code earlier, but the docstring still told users to run it
   via `~/workspace/hackathon/...`. Now generic.
4. **`README.md` — layout table was missing half the scripts**
   (`eval_policy.py`, `view_stairs.py`, `balls_of_solitude.py`,
   `download_g1_model.py`, `post_mortem.md`). Completed, and noted the
   cross-platform `.py` model downloader next to the `.sh` one.

## Deliberately not touched

- **Reward shaping / curriculum / training strategy.** The post-mortem and
  the team channel already converge on curriculum learning as the top bet
  for the stair trap (plus potential-based height shaping, reference-motion
  tracking, anti-hacking termination). Those change what gets trained and
  are a human call the morning of the demo — not a drive-by commit.
- **`jandre.txt` / `tony.txt`.** Team coordination files; not mine to edit.
- **No merge, no push.** Commits live on this branch only, for review.

## Suggested next steps (for the team, not this branch)

- Re-run `eval_policy.py --seeds 3` on the 2M checkpoint: with the seed fix,
  the deck number is now reproducible.
- If stairs stays the stretch goal, the queued shaping ideas in `jandre.txt`
  (21:12) are the highest-leverage change; the `--resume` fixes above make
  a re-run from the healthy 2M checkpoint safer than last night's attempt.
