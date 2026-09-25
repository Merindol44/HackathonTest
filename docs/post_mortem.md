# Stair-climbing PPO: overnight run post-mortem (2026-09-25)

Source: `runs/stairs_overnight/train.log` (frozen — pusher died with the run).
Env: 8 parallel MuJoCo G1 envs, SB3 PPO, 3e-4 LR. success_rate = 0 throughout.

## Run 1 (0 -> 1.47M steps; silent death ~21:20 CEST)

| timesteps | ep_rew_mean | ep_len_mean | approx_kl | clip_fraction | LR |
|----------:|------------:|------------:|----------:|--------------:|---:|
|    16,384 |       -53.2 |        60.5 |     0.000 |         0.000 | 3e-4 |
|   131,072 |       -37.5 |        65.9 |     0.174 |         0.686 | 3e-4 |
|   458,752 |        -6.1 |        57.6 |     0.534 |         0.797 | 3e-4 |
|   557,056 |        +0.2 |        59.8 |     0.632 |         0.811 | 3e-4 |
|   802,816 |        -0.3 |        59.8 |     0.931 |         0.839 | 3e-4 |
| 1,015,808 |       +10.6 |        60.4 |     1.111 |         0.854 | 3e-4 |
| 1,212,416 |       +17.2 |        65.7 |     1.275 |         0.864 | 3e-4 |
| 1,343,488 |       +22.8 |        66.6 |     1.526 |         0.870 | 3e-4 |
| 1,441,792 |       +26.1 |        68.4 |     1.611 |         0.875 | 3e-4 |
| 1,474,560 |       +26.9 |        66.5 |     1.611 |         0.876 | 3e-4 |

Clean climb: -53.2 -> +26.9. But ep_len stayed ~60-68 the whole run (flat)
and success_rate stayed 0 — the robot walks forward then falls at the stairs,
every episode. Dense forward-progress reward saturates the clip (~0.87):
the classic RL locomotion trap. KL crept 0 -> 1.61 — rising, but controlled.
**Death ~21:20 was silent** (last log line healthy): environmental, not instability.

## Run 2 (resume from 2M checkpoint ~21:40; death ~21:50 CEST)

| timesteps   | ep_rew_mean | ep_len_mean | approx_kl | clip_fraction | LR |
|------------:|------------:|------------:|----------:|--------------:|---:|
| (resume #1) 16,384  | -20.0 | 46.6 | 0.000 | 0.000 | 3e-4 |
| (resume #1) 32,768  | -19.7 | 51.9 | 4.231 | 0.926 | 3e-4 |
| (resume #1) 49,152  | -20.1 | 47.0 | 3.629 | 0.916 | 3e-4 |
| 2,016,384 |       -20.0 |        46.6 |     0.000 |         0.000 | 3e-4 |
| 2,032,768 |       -19.7 |        51.9 |     4.231 |         0.926 | 3e-4 |
| 2,049,152 |       -20.1 |        47.0 |     3.629 |         0.916 | 3e-4 |
| 2,065,536 |       -22.4 |        52.1 |     3.525 |         0.914 | 3e-4 |
| 2,081,920 |       -17.6 |        44.2 |     0.821 |         0.818 | 1.5e-4 |
| 2,098,304 |       -17.0 |        45.2 |     0.833 |         0.817 | 1.5e-4 |
| 2,131,072 |       -14.2 |        46.0 |     0.827 |         0.811 | 1.5e-4 |

What happened, in order:
1. **Resume #1 reset the timestep counter** (log shows 16k/32k/49k blocks
   after 1.47M) — this is what commit 36690f05 (`reset_num_timesteps=False`)
   fixed; without it we'd have overwritten run 1's checkpoints.
2. **The resume itself destabilized a healthy policy.** Pre-crash: +27 reward,
   KL 1.61. First post-resume blocks: reward -20, KL 4.23 by the second block.
   The VecNormalize restore was fine — the *policy* destabilized (PPO resuming
   from a checkpoint is not guaranteed-stable).
3. **The KL watchdog fired exactly as designed**: sustained approx_kl > 2.0 for
   3 rollouts -> emergency checkpoint (21:45:49) + LR 3e-4 -> 1.5e-4, schedule
   pinned. KL fell back to ~0.82 — then the process died anyway ~21:50.
4. **Cause of the 21:50 death is unknown**: no OOM/killed-process lines in
   dmesg/journal (container — a host-level OOM kill may not show here), RAM was
   3.0G/7.7G free. Watch RAM on any re-run.

## Decisions

- NO restart overnight. Two deaths in ~35 min + a destabilized resume.
- Morning: render final video + 3-seed eval off the **2M checkpoint**
  (`ppo_stairs_2000000_steps.zip`, healthy +27) — never the
  `kl_watchdog_checkpoint` (saved mid-instability, by design).
- The 2M checkpoint is an honest result: 250k-vs-2M learning curve for the deck.
