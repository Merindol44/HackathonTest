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
| 02:40–03:30 | `night_v1b_flat` | v1 flat + gait tracking (w=1.0) + anti-stand, from scratch | ~250k/400k (stopped) | ~93 | ~79 | 0 | — | **Stopped early — gait reference infeasible.** Open-loop test: hand-designed 1.8 Hz gait falls at 98 steps; 12-variant search found only a *backward*-walking gait (201 steps, −1.04 m); no forward gait survived. Policy was right not to track it (leg MSE 0.07). Pivoting to fall-harness ("training wheels") curriculum instead. |
| 03:40–04:15 | `night_h1_flat` | v1 flat + harness=1.0 + anti-stand, from scratch | 303104/300k | +163 | 523 | 0.05 | — | **Completed. BREAKTHROUGH.** Episodes 6× longer (523 vs ~80), reward −3.5→+163, and the first non-zero success_rate of the night (5% walk 6 m). Rollout probe: policy walks 2.7–4.4 m upright (pelvis_z ~0.72) with harness. The training-wheels curriculum works — PPO discovers walking when episodes are long. Annealing harness next. |
| 04:18–04:55 | `night_h2_flat` | v1 flat + harness=0.5, resumed from H1 | 603104 (300k new) | +323 | 627 | 0.02 | 0.56 | **Completed.** Reward +163→+323 as harness halved — the policy supports its own weight more (less harness penalty) and walks better. Annealing continues. |
| 04:58–05:35 | `night_h3_flat` | v1 flat + harness=0.2, resumed from H2 | 906208 (300k new) | +96.2 | 405 | 0 | 0.58 | **Completed.** Expected annealing dip (+323→+42) then recovery (+106, ep_len 251→405). Policy adapts to reduced support. Final anneal to zero next. |
| 05:38–06:05 | `night_h4_flat` | v1 flat + harness=0.0, resumed from H3 | 934880 (died, 28k new) | +15 | 55 | 0 | 3.75 | **Died after KL explosion** (3.75 > 2.0, watchdog halved LR then silent death). No recovery over 28k steps — the 0.2→0.0 jump is too big; policy reverted to 55-step flailing. **Not restarted identically** (diagnosed config gap, not a fluke). Pivoting: take the good H3 walker (harness=0.2) to stairs instead, continue annealing there. |
| 06:12–06:55 | `night_h4b_stairs` | 3×0.06 m stairs + harness=0.2, resumed from H3 | 1209312 (300k new) | +287 | 410 | 0 | 0.93 | **Completed.** KL watchdog fired once (3.9, stair transition) but run recovered. Policy reaches stairs (x=1.7 m in probes) but **levels=0 — never steps up**. Root cause found: level bonus used *pelvis* height (requires lifting whole body), not *foot* height. Fixed to foot-based; testing in H5. |
| ~05:45 | — | — | — | — | — | — | — | **Infra note:** runtime service restarted, wiping `/tmp` (supervisor script + state). H4 training survived (separate process). Rewrote supervisor from scratch, fixed a resume bug it had (`--timesteps` on restart now passes only the *remaining* steps to TARGET), re-armed for `night_h4_flat`. No training data lost; branch was already pushed. |
| ~06:10 | — | — | — | — | — | — | — | **Infra note 2:** Python packages wiped again by the restart (`ModuleNotFoundError`). Reinstalled identical versions (torch 2.14.0+cpu, mujoco 3.14.0, gymnasium 1.3.0, sb3 2.9.0) with `--break-system-packages`. H4b relaunched cleanly afterwards. |

## Checkpoints
- Best final checkpoint + vecnormalize pkl will be force-added here at the end.
