# Learning methods for industrial humanoids — comparison & recommendation

*SKF hackathon challenge: "Analyze and compare learning methods suitable for
industrial humanoids". This document is the analysis half of our deliverable.
It is grounded in the pipeline we actually built: a Unitree G1 humanoid
learning stair climbing in MuJoCo with PPO (`scripts/g1_stairs_env.py`,
`scripts/train_stairs.py`).*

## 1. The core question

> How can we make the training of humanoids in an industrial setting easier
> and more efficient?

No single learning method wins on every industrial requirement. The honest
answer — and our recommendation — is a **hybrid pipeline**: use cheap,
safe methods where they suffice, and expensive, powerful methods only where
nothing else works.

## 2. Methods compared

| Method | Data effort | Training time | Safety | Repeatability | Adaptability | Compute | Explainability | Deployment maturity |
|---|---|---|---|---|---|---|---|---|
| **Manual programming** (trajectory scripting, MPC) | None | Hours–days of engineer time | ★★★★★ (fully predictable) | ★★★★★ | ★☆☆☆☆ (recode per change) | Tiny | ★★★★★ | ★★★★★ (industry standard today) |
| **Teleoperation / learning from demonstration** | Medium (hours of expert demos) | Hours | ★★★★☆ (human in loop) | ★★★☆☆ | ★★★☆☆ | Small | ★★★☆☆ | ★★★★☆ (used in warehouses) |
| **Imitation learning** (behavior cloning, diffusion policies) | High (100s–1000s of demos) | Hours–days | ★★★☆☆ | ★★★☆☆ (covariate shift) | ★★☆☆☆ | Medium (GPU) | ★★☆☆☆ | ★★★☆☆ (growing fast) |
| **Reinforcement learning** (PPO, in sim) | None (self-generated) | Days–weeks sim time | ★★☆☆☆ in sim → safe; real-world RL unsafe | ★★★★☆ (converges to a policy) | ★★★★☆ | Large (GPU cluster) | ★☆☆☆☆ | ★★☆☆☆ (few real deployments) |
| **Sim-to-real transfer** (domain randomization) | None extra | Adds 20–50% training | ★★★★☆ (validated in sim first) | ★★★★☆ | ★★★★☆ | Large | ★★☆☆☆ | ★★★☆☆ (works for locomotion) |
| **Human feedback** (RLHF / corrective interventions) | Medium (human labels) | Days | ★★★★☆ | ★★★☆☆ | ★★★★☆ | Medium | ★★★☆☆ | ★★☆☆☆ (research stage) |
| **VLA / foundation models** (RT-2, π0-style) | Huge (pretraining; fine-tune small) | Minutes–hours to adapt | ★★☆☆☆ (black box) | ★★☆☆☆ | ★★★★★ | Very large | ★☆☆☆☆ | ★☆☆☆☆ (not industrial yet) |

*Ratings are qualitative, calibrated for industrial deployment at SKF-like
facilities.*

## 3. Locomotion vs manipulation vs complete workflows

**Locomotion (our task: stair climbing).**
Locomotion is a *dense-reward, physics-dominated* problem: the objective
("move forward without falling") is easy to specify mathematically, and
contact dynamics are hard for humans to demonstrate well. This is RL's home
turf. Our pipeline proves the point: with a shaped reward (forward velocity +
upward velocity − energy − tilt), PPO learns from scratch in simulation with
**zero human demonstration data**. Sim-to-real via domain randomization
(mass, friction, latency, terrain variation) is the established bridge to the
real G1 — the same recipe that put RL walkers on real quadrupeds years ago.

**Manipulation (pick-and-place, wiping, tool use).**
Manipulation inverts the trade-off: objectives are easy for humans to show
("grasp *this* handle") but painful to encode as reward functions (what is
the reward for a "good wipe"?). Pure RL struggles with sparse rewards and
unsafe exploration around fragile parts. **Imitation learning from
teleoperation wins here**: collect 100–300 demos with a leader–follower rig,
train a diffusion policy, deploy. Data effort is real but bounded, and the
resulting behavior looks human-sensible — important for working next to
people.

**Complete industrial workflows (locomotion + manipulation + task logic).**
No single method covers a full workflow. The realistic architecture is
**hierarchical and hybrid**:
- *Task planner* (rules / state machine, or a VLA model for open-ended
  commands like "fetch the bearing from shelf B") decides *what* to do.
- *Locomotion module*: RL policy (sim-to-real), runs continuously.
- *Manipulation module*: imitation-learned skills, triggered by the planner.
- *Safety layer*: manual programming — hard-coded interlocks, speed limits,
  and e-stops. Never learned, always on.

## 4. Where hybrid approaches win (our recommendation for SKF)

| Workflow stage | Recommended method | Why |
|---|---|---|
| Locomotion skills (walk, stairs, step-length) | **RL in simulation + sim-to-real** | No demo data needed; our G1 pipeline demonstrates feasibility |
| Manipulation skills (grasp, place, wipe) | **Teleoperation → imitation learning** | Human demos beat hand-shaped rewards for contact-rich tasks |
| Task sequencing & error recovery | **Manual programming / state machines** | Deterministic, auditable, certifiable |
| Adapting to new parts / layouts | **Foundation-model / VLA fine-tuning** | Language-conditioned generalization without re-engineering |
| Safety envelope | **Manual programming (interlocks)** | Learned components must never override hard limits |

**Business read:** locomotion via RL is the cheapest skill to acquire (compute
is cheaper than expert demo time); manipulation via imitation has the fastest
time-to-first-deployment; the safety layer is non-negotiable for insurance and
certification. A hybrid stack lets SKF deploy incrementally — one skill at a
time — instead of betting the factory on a single paradigm.

## 5. What our build demonstrates (and its limits)

*What works:* the full loop — MuJoCo G1 model → Gymnasium env with shaped
rewards → PPO → checkpointed policy → rendered video — runs on commodity
hardware, no robot required. This is the "try it in simulation first" step
the brief asks for.

*Known limits / honest caveats:*
- **Sim-to-real gap**: our policy is trained on one staircase geometry with
  no domain randomization yet; real stairs vary in height, friction, and
  lighting. Adding randomization is the explicit next step before hardware.
- **Sample efficiency**: 3M steps on CPU is a *feasibility* run, not a
  converged champion. With SKF's Cloud GPUs and vectorized envs, 50–100M
  steps (the regime where humanoid locomotion really converges) become
  practical.
- **No vision**: our observation is proprioceptive only. Real stair climbing
  needs terrain sensing (depth/vision) — the natural upgrade is adding a
  heightmap to the observation and randomizing stair geometry per episode.
- **Explainability**: the PPO policy is a black box. For deployment, SKF
  should wrap it in the manual safety layer described above.

## 6. Bottom line for the judges

1. **RL is the right tool for locomotion** — we built it, it runs, video
   below.
2. **Imitation is the right tool for manipulation** — analyzed, not built
   (out of scope for one hackathon night, honest about it).
3. **The industrial answer is hybrid**: RL locomotion + imitation
   manipulation + programmed safety, sequenced by a task planner.
4. **Path to the real G1**: domain randomization → hardware validation on
   flat ground → low stairs → full staircase, with the safety layer active
   from step one.
