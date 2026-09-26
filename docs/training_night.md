# Overnight training log — 2026-09-26 (branch `muse/review-improvements`)

<!-- LIVE-STATUS-START -->
## Live status (auto-updated)

- **Updated:** 03:52 UTC (05:52 CEST)
- **Run:** `night_h13_full` (continuation coordinator)
- **State:** RUNNING — 6×0.12 m stairs (original target geometry), harness=0.2, r_level=5.0 (300k steps, resumed from H11 1.86M)
- **Timesteps:** 1864672 → +300k planned
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
| 06:58–07:22 | `night_h5_stairs` | 3×0.06 m stairs + harness=0.2 + **foot-based level bonus**, resumed from H4b | 1410016 (200k new) | +159 | 282 | 0 | — | **Completed. FIRST STAIR CLIMBS.** Probe: 3/5 episodes achieve **levels=1** (foot on first 0.06 m step). The reward fix unlocked stepping-up. Reward dipped (+287→+159) as policy explores the new skill. Consolidating in H6. |
| 07:23–07:38 | `night_h6_stairs` | 3×0.06 m stairs + harness=0.2, resumed from H5 | 1561568 (150k new) | +219 | 350 | 0 | 1.51 | **Completed (final training run).** 10-episode probe: **max levels=2** (two 0.06 m steps climbed), 4/10 episodes reach levels≥1, mean levels 0.50. **Harness=0.0 probe: total failure** (41–84 steps, 0.00–0.15 m, levels=0) — policy is harness-dependent; annealing incomplete. Best checkpoint: `runs/night_h6_stairs/final_model.zip`. |
| 08:04–08:15 | `night_h7_stairs` | 3×0.06 m stairs + harness=0.15, resumed from H6 (continuation coordinator) | 1762272 (200k new) | +87.9 | 169 | 0 | 1.77 | **Completed. Annealing dip.** KL watchdog fired once at resume start (harness-change shock, LR 3e-4→1.5e-4) then KL stayed healthy (1.77). Reward +219→+88, ep_len 350→169. Probe @0.15: max levels=2, mean 0.2, 1/10 reach ≥1, mean_max_x=0.4 m — walking regressed. Probe @0.0: still total failure (levels=0, 0.05 m). No KL explosion; reward trending up at end (+77.6→+87.9). Continuing anneal to 0.10. |
| 08:18–08:29 | `night_h8_stairs` | 3×0.06 m stairs + harness=0.10, resumed from H7 | 1962976 (200k new) | +52.2 | 91.7 | 0 | 0.50 | **Completed. Walker dying.** KL healthy (0.50, no explosion; watchdog fired twice at resume, LR→7.5e-05). But reward +88→+52, ep_len 169→92, and probe @0.10 shows **levels=0 everywhere, mean_max_x=0.22 m** — policy can no longer walk forward to the stairs. Probe @0.0: still failure. Stepping down blindly would kill the walker; running a **consolidation stage at harness=0.10** (H9, 300k) before any further anneal. |
| 08:31–08:47 | `night_h9_stairs` | 3×0.06 m stairs + harness=0.10 (consolidation), resumed from H8 | 2266080 (300k new) | +60.2 | 91.6 | 0 | 0.52 | **Completed. Consolidation FAILED to recover the walker.** Reward +52→+60 (marginal), ep_len flat (91.7→91.6). Probe @0.10: max_levels=1 (one lucky ep), mean 0.1, mean_max_x=0.3 m — still cannot walk to stairs. 300k extra steps at 0.10 did not rebuild walking. **Pivot:** stairs are a harder locomotion task than the broken walker can handle; moving recovery to **flat ground** (H10, harness=0.10, 200k). If flat walking recovers, anneal 0.05→0.0 on flat, then reintroduce stairs. |
| 08:49–09:00 | `night_h10_flat` | flat + harness=0.10, resumed from H9 | 2466784 (200k new) | +60.7 | 95.2 | 0 | 2.75 | **Completed. Flat recovery FAILED too.** ep_len 91.6→95.2 (no recovery), KL elevated 2.6–2.8 sustained (flat domain shift destabilized updates). Probe @0.10: mean_max_x=0.28 m — walker not rebuilt on flat either. Probe distributions uniformly bad (not bimodal): the walking skill is effectively gone at harness≤0.10. **Verdict: the harness anneal has stalled at ~0.10.** Restoring last-good checkpoint (H6, harness=0.2) and pivoting to the strongest possible harness-assisted climber: R_LEVEL 2.0→5.0 boost (recommended experiment #3), H11 300k. |
| 09:01–09:18 | `night_h11_climb` | 3×0.06 m stairs + harness=0.2 + **R_LEVEL 2.0→5.0** (new `--r-level` flag), resumed from H6 | 1864672 (300k new) | +271 | 372 | 0 | 0.49 | **Completed. Stronger climber.** Reward +219→+271, KL healthy. Probe @0.2: **mean_levels 0.9 (was 0.5), 7/10 reach ≥1 level** (was 4/10), max 2/3, mean_max_x=1.19 m. Probe @0.0: still total failure. R_LEVEL boost helped but no full 3-level climbs yet. **New idea for the anneal wall:** discrete harness drops shock the policy (KL spike → LR halving → walker regression). Implementing **smooth within-run harness annealing** (`set_harness` + `HarnessAnnealCallback`, `--harness-end/--harness-anneal-steps`) and testing 0.2→0.0 over 800k steps on flat (H12). |
| 09:24–10:07 | `night_h12_anneal` | flat + **smooth harness anneal 0.2→0.0** over 800k new steps, resumed from H6 | 2364384 (800k new) | +63.8 | 78 | 0 | 0.59 | **Completed. Smooth anneal ALSO FAILED.** Harness ramped 0.2→0.0 cleanly (KL healthy throughout, no explosions). But the walker still died: ep_len 78 at harness≈0. Probe @0.0 (flat AND stairs): levels=0, mean_max_x=0.23 m. **Diagnostic:** mid-anneal checkpoints show the walker was already dead by harness≈0.125 (300k into the ramp); H6 probed on flat @0.2 walks 2.03 m fine, so it was the anneal — not a domain shift — that killed it. **Conclusion: the policy cannot learn unsupported balance through harness annealing (discrete or smooth). The harness was doing the balancing; PPO never learned a balance controller.** Unsupported walking/climbing NOT achieved tonight. Pivoting remaining budget to the best harness-assisted result: the original 6×0.12 m target geometry at harness=0.2 (H13). |
| ~05:45 | — | — | — | — | — | — | — | **Infra note:** runtime service restarted, wiping `/tmp` (supervisor script + state). H4 training survived (separate process). Rewrote supervisor from scratch, fixed a resume bug it had (`--timesteps` on restart now passes only the *remaining* steps to TARGET), re-armed for `night_h4_flat`. No training data lost; branch was already pushed. |
| ~06:10 | — | — | — | — | — | — | — | **Infra note 2:** Python packages wiped again by the restart (`ModuleNotFoundError`). Reinstalled identical versions (torch 2.14.0+cpu, mujoco 3.14.0, gymnasium 1.3.0, sb3 2.9.0) with `--break-system-packages`. H4b relaunched cleanly afterwards. |

## Checkpoints
- Best final checkpoint + vecnormalize pkl will be force-added here at the end.

**01:52 UTC — `night_h6_stairs` COMPLETED** at 1561568 steps.

## Night verdict (07:45 CEST)
Harness curriculum (1.0→0.5→0.2) taught the G1 to walk; a **foot-based level-bonus fix** unlocked stair stepping (max **2 levels** of 3×0.06 m with harness=0.2). **Unsupported (harness=0.0) walking/stair-climbing was NOT achieved** — the 0.2→0.0 anneal step collapsed (KL explosion) and the policy remains harness-dependent. The final 6×0.12 m target was not attempted. Physics fidelity: gravity −9.81, 33.34 kg, 2 ms timestep verified; but all successful locomotion used external harness support (not real physics).

## Recommended next experiment
1. **Gentler harness anneal on stairs:** 0.2→0.15→0.1→0.05→0.0 with 200k steps each, starting from H6. The 0.2→0.0 jump was too big; intermediate steps may bridge it.
2. **Apply the other two v1 reward fixes** (velocity band strictly zero outside [0.25,1.0]; per-foot cadence with 0.08 m arm / 0.03 m strike / 30 cap) — they were never implemented.
3. **Increase R_LEVEL** (2.0→5.0) to strengthen the stair-climbing signal now that the bonus is foot-based and achievable.
4. Then 6×0.12 m stairs, harness=0.0, for the final real-physics validation.
