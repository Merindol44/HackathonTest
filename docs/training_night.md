# Overnight training log — 2026-09-26 (branch `muse/review-improvements`)

<!-- LIVE-STATUS-START -->
## Live status (auto-updated)

- **Updated:** — (supervisor starting)
- **Run:** `night_baseline`
- **State:** TRAINING
- **Timesteps:** —
- **ep_rew_mean:** — | **ep_len_mean:** — | **success_rate:** — | **approx_kl:** —
- **Restarts this run:** 0
<!-- LIVE-STATUS-END -->

**Orders in effect:** (1) auto-restart any dead run from its latest checkpoint
via `--resume` (max 3 restarts, then abandon + move on); (2) this file is
refreshed at least every 30 min (immediately on restarts/completions) for the
hourly user briefing.

Coordinator: Muse (overnight subagent). Physics verified full-fidelity at start:
`model.opt.gravity = [0, 0, -9.81]`, total body mass 33.34 kg, contacts live
(ncon=7 standing), dt=0.002. No video rendering this night — numbers only.

Box: 2 vCPU / ~2 GB free RAM, torch CPU. 4 envs (SubprocVecEnv), PPO defaults
from `scripts/train_stairs.py` unless noted. All runs headless.

## Plan
1. `night_baseline` — v0 env repro, 200k steps (sanity + comparison point).
2. `night_v1_flat` — v1 rewards, flat ground (0 stairs), 300k from scratch.
3. `night_v1_3step` — v1 rewards, 3×0.06 m, 300k, `--resume` from stage 2.
4. `night_v1_full` — v1 rewards, 6×0.12 m, `--resume` from stage 3, remaining budget.
5. If time: ablation (v1 from scratch on full stairs, or ent_coef=0.01 variant).

## Results

| Time (CEST) | Run | Variant | Steps | ep_rew_mean | ep_len_mean | success_rate | approx_kl | Notes |
|---|---|---|---|---|---|---|---|---|
| 00:42–00:52 | `night_baseline` | v0, 4 envs | 147k/200k | −20.4 | 63.8 | 0 | 0.30 | **Run died silently at 147k** (no traceback, log frozen 22:49 UTC) — same signature as the 21:20 death in the post-mortem. No OOM lines visible (container). 100k checkpoint intact. Reward was climbing −47→−20, learning confirmed. |
| ~01:00 | — | — | — | — | — | — | — | **Environment wiped**: `/usr/local/lib/python3.12/dist-packages` was emptied by an external event (torch/mujoco/gymnasium/sb3 all gone; the training run was already dead). Reinstalled from pip cache in ~1 min: torch 2.14.0+cpu, mujoco 3.14.0, gymnasium 1.3.0, sb3 2.9.0 — identical versions. Relaunching baseline with 2 envs (RAM caution). |
| 01:08–01:13 | `night_baseline` | v0, 2 envs | 200704/200k | −8.74 | 64.0 | 0 | 0.66 | **Completed.** Reward −47→−8.7, clean learning curve on 2 envs, no instability. success_rate 0 (expected — v0 locomotion trap). This is the comparison point for v1/curriculum. |
| 01:15–01:42 | `night_v1_flat` | v1, flat (0 stairs), from scratch | 401408/400k | +87.9 | 81.2 | 0 | 1.28 | **Completed.** v1 band rewards learn much faster (+20 at 50k). success_rate 0 — flat success (walk 6 m) not reached; policy walks briefly then falls. Still a walking foundation for the curriculum. KL 1.28, watch. |
| 01:45–02:35 | `night_v1_3step` | v1, 3×0.06 m, resumed from Stage A 400k | 854656 (454k new) | +121 | 97.7 | 0 | 0.63 | **Stopped at intended 400k of new training** (SB3 resume semantics: `--timesteps` is *additional* on resume — noted for future stages). Reward +88→+121, KL watchdog halved LR once (3e-4→1.5e-4, healthy). success_rate 0 — rollout probe shows the robot never reaches the stairs (max x 0.16–0.54 m vs stairs at 1.2 m). Pure curriculum cannot fix the lunge-and-fall local optimum. Pivoting to gait-tracking prior. |

## Checkpoints
- Best final checkpoint + vecnormalize pkl will be force-added here at the end.
