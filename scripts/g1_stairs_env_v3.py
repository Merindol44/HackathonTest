"""G1 stair-climbing env v2: reward v3 ("all eight ideas").

Same 73-dim proprioceptive observation and 29-dim action space as v1, same
parameterized geometry (n_stairs, step_h). Reward v3 keeps the v1 terms that
worked (velocity band, clearance, knee lift, anti-dodge, alive, energy
penalty, harness-force penalty, fall/success terminals, anti-stand) and
adds all eight brainstorm ideas:

1. Potential-based dense progress: every-step shaping from the distance to
   the next uncleared step (gamma*Phi' - Phi, policy-invariant).
2. Per-foot ALTERNATING cadence: a foot-strike pays only if the other foot
   struck last AND the pelvis moved forward since (no stomping in place).
3. Single-support balance: pelvis-over-stance-foot reward during single
   support (explicit balance objective, harness-independent).
4. Split level bonus: half when the first foot crosses a level, half when
   the second foot joins it (no straddling).
5. Harness BUDGET: support fades linearly to zero over HARNESS_BUDGET_STEPS
   within each episode (wean inside the run, not across runs).
6. Foot placement: swing-foot landing rewarded near the next step's center.
7. Landing softness: penalty on foot impact velocity at touchdown.
8. Pelvis height TRACKING: target pelvis height rises step_h per fully
   cleared level (replaces the bouncy r_up velocity bonus).

Run from scripts/:  python -c "from g1_stairs_env_v3 import G1StairsEnvV3; ..."
"""

import numpy as np

from g1_stairs_env import (
    ACTION_SCALE,
    FALL_Z,
    MAX_EPISODE_STEPS,
    N_ACT,
    N_SUBSTEPS,
    OBS_DIM,
    PHYS_DT,
    PLAT_LEN,
    R_FALL,
    R_SUCCESS,
    STAIR_W,
    STEP_D,
    TILT_LIM,
    W_ALIVE,
    W_ENERGY,
    X0,
    G1StairsEnv,
    _quat_to_euler,
)

import mujoco
from gymnasium import spaces

# v3 reward weights (v1 terms kept unless noted)
W_BAND = 1.5      # forward-velocity band
W_CLEAR = 1.0     # foot clearance near stairs
W_KNEE = 0.5      # knee lift near stairs
R_LEVEL = 2.0     # per new stair level reached (split half/half per foot)
W_VY = 0.4        # anti side-dodge
W_Y = 0.2

# Idea 1: potential-based dense progress toward the next uncleared step.
POT_WX = 2.0      # weight on horizontal distance to next step edge
POT_WZ = 4.0      # weight on vertical gap to next step top
POT_GAMMA = 0.99  # must match PPO gamma for policy invariance

# Idea 2: per-foot alternating cadence (replaces the farmable r_strike).
R_ALT = 0.30            # per alternating strike (with forward progress)
MAX_ALT_STRIKES = 60    # cap per episode
MIN_STRIKE_DX = 0.05    # min pelvis forward progress since last strike (m)

# Idea 3: single-support balance (pelvis over stance foot).
CONTACT_Z = 0.025  # foot height at/below this counts as ground contact
W_BAL = 1.0
BAL_K = 40.0       # exp kernel on horizontal pelvis-stancefoot distance^2

# Idea 6: foot placement near next step center (on landing).
W_PLACE = 0.5
PLACE_K = 30.0

# Idea 7: landing softness (impact velocity penalty at touchdown).
W_IMPACT = 0.10
IMPACT_VMAX = 3.0  # clip impact speed (m/s)

# Idea 8: pelvis height tracking (replaces r_up).
W_HZ = 1.0
HZ_K = 25.0

# Torso posture (v4): reward an upright torso with a slight forward lean.
# Replaces the old quadratic tilt penalty around zero pitch — the optimum
# is now a small forward pitch, not exactly zero (positive pitch = forward).
W_POSTURE = 1.0
POSTURE_K = 12.0
PITCH_TGT = 0.08   # ~4.6 deg forward lean ("very tiny bit")

# Idea 5: harness budget — support fades linearly to zero within each
# episode over this many steps (wean inside the run, not across runs).
HARNESS_BUDGET_STEPS = 700

# Fall-harness (training wheels): a virtual support that holds the pelvis
# upright at standing height while the policy learns leg control. The policy
# is penalized for the support force used, so it is incentivized to support
# its own weight; anneal `harness` to 0 for full physics.
HARNESS_ZTGT = 0.78   # target pelvis height
HARNESS_KP = 8000.0
HARNESS_KD = 400.0
HARNESS_FMAX = 800.0
HARNESS_TILT_KP = 400.0   # righting moment gains (torso stabilization)
HARNESS_TILT_KD = 40.0
HARNESS_MMAX = 300.0
HARNESS_W = 0.002      # penalty per Newton of harness force used
HARNESS_MW = 0.004     # penalty per Nm of harness moment used

# Reference-gait tracking (imitation prior for the hard-to-discover walking
# pattern). Open-loop phase; the task rewards still select real locomotion.
GAIT_FREQ = 1.8          # Hz, human-like cadence
GAIT_HIP_AMP = 0.35      # rad
GAIT_KNEE_BASE, GAIT_KNEE_AMP = 0.15, 0.50
TRACK_K = 8.0            # exp kernel sharpness on leg-joint MSE
LEG_JNTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
ANTI_STAND_STEPS = 120   # truncate if no forward progress in this many steps
ANTI_STAND_DX = 0.02

VX_LO, VX_HI = 0.25, 1.0     # realistic walking band (m/s)
VX_MAX = 1.5                # hard ramp-down end
STRIKE_HIGH, STRIKE_LOW = 0.08, 0.03   # foot-strike detection (m)


def _band_vx(vx: float) -> float:
    """Trapezoid: 0 below 0, ramp 0->0.25, 1.0 on [0.25, 1.0], ramp down to 0 at 1.5."""
    if vx <= 0.0:
        return 0.0
    if vx < VX_LO:
        return vx / VX_LO
    if vx <= VX_HI:
        return 1.0
    if vx < VX_MAX:
        return (VX_MAX - vx) / (VX_MAX - VX_HI)
    return 0.0


def _build_model_v1(g1_xml_path, n_stairs, step_h):
    """Compose G1 + ground + n_stairs steps + top platform via MjSpec."""
    import mujoco

    spec = mujoco.MjSpec()
    loaded = spec.from_file(str(g1_xml_path))
    if loaded is not None:
        spec = loaded
    world = spec.worldbody

    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[8.0, 8.0, 0.1],
        rgba=[0.92, 0.92, 0.94, 1.0],
    )
    top_h = n_stairs * step_h
    for i in range(n_stairs):
        h = (i + 1) * step_h
        world.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[STEP_D / 2, STAIR_W / 2, h / 2],
            pos=[X0 + STEP_D / 2 + i * STEP_D, 0.0, h / 2],
            rgba=[0.45, 0.47, 0.55, 1.0],
        )
    if n_stairs > 0:
        world.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[PLAT_LEN / 2, STAIR_W / 2, top_h / 2],
            pos=[X0 + n_stairs * STEP_D + PLAT_LEN / 2, 0.0, top_h / 2],
            rgba=[0.30, 0.55, 0.35, 1.0],
        )
    return spec.compile()


class G1StairsEnvV3(G1StairsEnv):
    """Curriculum-capable stairs env with v3 shaped rewards (all 8 ideas).

    Args:
        n_stairs: number of steps (0 = flat ground, walking task).
        step_h: height of each step (m).
    """

    def __init__(self, g1_xml=None, n_stairs=6, step_h=0.12, render_mode=None,
                 track_w=0.0, anti_stand=False, harness=0.0, r_level=2.0):
        # Bypass G1StairsEnv.__init__ (it hardcodes v0 geometry) and repeat
        # the setup with our builder. Observation/action spaces identical.
        import gymnasium as gym
        from pathlib import Path

        gym.Env.__init__(self)
        if g1_xml is None:
            g1_xml = Path(__file__).resolve().parent.parent / "assets" / \
                "mujoco_menagerie" / "unitree_g1" / "g1.xml"
        self.n_stairs = int(n_stairs)
        self.step_h = float(step_h)
        self.track_w = float(track_w)
        self.anti_stand = bool(anti_stand)
        self.harness = float(harness)
        self.r_level = float(r_level)  # per new stair level reached (R_LEVEL default)
        self.model = _build_model_v1(Path(g1_xml), self.n_stairs, self.step_h)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_ACT,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        _k = mujoco.mjtObj.mjOBJ_KEY
        self._stand_qpos = self.model.key_qpos[
            mujoco.mj_name2id(self.model, _k, "stand")
        ].copy()

        jnt = self.model.actuator_trnid[:, 0].astype(int)
        self._qpos_adr = self.model.jnt_qposadr[jnt]
        self._dof_adr = self.model.jnt_dofadr[jnt]
        lo = self.model.jnt_range[jnt, 0]
        hi = self.model.jnt_range[jnt, 1]
        self._jnt_lo = lo
        self._jnt_hi = hi
        self._jnt_mid = (lo + hi) / 2.0
        self._jnt_span = (hi - lo) / 2.0
        self._stand_act = self._stand_qpos[7:].copy()

        _b = mujoco.mjtObj.mjOBJ_BODY
        self._pelvis = mujoco.mj_name2id(self.model, _b, "pelvis")
        self._lfoot = mujoco.mj_name2id(self.model, _b, "left_ankle_roll_link")
        self._rfoot = mujoco.mj_name2id(self.model, _b, "right_ankle_roll_link")
        self._lknee = mujoco.mj_name2id(self.model, _b, "left_knee_link")
        self._rknee = mujoco.mj_name2id(self.model, _b, "right_knee_link")

        if self.n_stairs > 0:
            self._success_x = X0 + self.n_stairs * STEP_D + 0.6
        else:
            self._success_x = 6.0  # flat stage: walk far without falling

        self._steps = 0
        self.render_mode = render_mode
        self._renderer = None
        # Per-episode shaping state (set in reset()).
        self._base_pelvis_z = 0.79
        self._levels = {}          # stair level k -> set of feet {'l','r'} that crossed
        self._prev_phi = 0.0       # idea 1: previous potential
        self._last_strike_foot = None  # idea 2: which foot struck last
        self._last_strike_x = 0.0      # idea 2: pelvis x at last strike
        self._alt_strikes = 0          # idea 2: alternating strikes this episode
        self._prev_lfoot_z = 0.0   # ideas 2/7: per-foot previous heights
        self._prev_rfoot_z = 0.0
        self._phase = 0.0
        self._best_x = 0.0
        self._last_prog_step = 0

    def set_harness(self, value: float):
        """Adjust the fall-harness strength mid-run (for smooth annealing).

        The harness force is recomputed from ``self.harness`` every step, so
        this takes effect immediately on all live envs.
        """
        self.harness = float(max(0.0, min(1.0, value)))

    # ------------------------------------------------------------------
    def _gait_ref(self):
        """Reference joint angles for the 12 leg DOFs at the current phase."""
        ref = np.zeros(12)
        for side, ph in ((0, self._phase), (6, self._phase + np.pi)):
            s, c = np.sin(ph), np.cos(ph)
            hip = GAIT_HIP_AMP * s
            knee = GAIT_KNEE_BASE + GAIT_KNEE_AMP * max(0.0, c)
            ankle = -(0.5 * hip + 0.4 * (knee - GAIT_KNEE_BASE))
            ref[side + 0] = hip    # hip_pitch
            ref[side + 3] = knee   # knee
            ref[side + 4] = ankle  # ankle_pitch
            # hip_roll / hip_yaw / ankle_roll stay at 0
        return ref

    # ------------------------------------------------------------------
    def _next_level(self) -> int:
        """Smallest stair level not yet fully cleared (both feet)."""
        for k in range(1, self.n_stairs + 1):
            if set(self._levels.get(k, ())) != {"l", "r"}:
                return k
        return self.n_stairs + 1

    def _potential(self, pelvis_x: float, max_foot_z: float) -> float:
        """Idea 1: potential pulling toward the next uncleared step.

        Phi = -(POT_WX * horizontal gap to next step edge
                + POT_WZ * vertical gap from best foot to step top).
        Paid as gamma*Phi' - Phi each step (policy-invariant shaping).
        """
        k = self._next_level()
        if k <= self.n_stairs:
            x_edge = X0 + k * STEP_D
            z_top = k * self.step_h
        else:  # all steps cleared: pull toward the top platform exit
            x_edge = X0 + self.n_stairs * STEP_D + 0.6
            z_top = self.n_stairs * self.step_h
        dx = max(0.0, x_edge - pelvis_x)
        dz = max(0.0, z_top - max_foot_z)
        return -(POT_WX * dx + POT_WZ * dz)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Baseline pelvis height for level thresholds (measured, not assumed).
        self._base_pelvis_z = float(self.data.xpos[self._pelvis][2])
        self._levels = {}
        self._last_strike_foot = None
        self._last_strike_x = float(self.data.xpos[self._pelvis][0])
        self._alt_strikes = 0
        self._prev_lfoot_z = float(self.data.xpos[self._lfoot][2])
        self._prev_rfoot_z = float(self.data.xpos[self._rfoot][2])
        self._prev_phi = self._potential(
            float(self.data.xpos[self._pelvis][0]),
            max(self._prev_lfoot_z, self._prev_rfoot_z),
        )
        if self.n_stairs > 0:
            self._success_z = self._base_pelvis_z + self.n_stairs * self.step_h - 0.15
        else:
            self._success_z = 0.5
        self._phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        self._best_x = float(self.data.xpos[self._pelvis][0])
        self._last_prog_step = 0
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self._stand_act + action * (ACTION_SCALE * self._jnt_span)
        target = np.clip(target, self._jnt_lo, self._jnt_hi)
        d = self.data
        # Idea 5: harness budget — support fades linearly to zero within each
        # episode, forcing the policy to wean itself off inside the run.
        eff_harness = self.harness * max(
            0.0, 1.0 - self._steps / HARNESS_BUDGET_STEPS
        )
        for _ in range(N_SUBSTEPS):
            d.ctrl[:] = target
            if eff_harness > 0.0:
                pz = d.xpos[self._pelvis][2]
                vz = d.qvel[2]
                fz = eff_harness * (HARNESS_KP * (HARNESS_ZTGT - pz)
                                    - HARNESS_KD * vz)
                d.xfrc_applied[self._pelvis, 2] = float(
                    np.clip(fz, 0.0, HARNESS_FMAX))
                # Righting moments on the torso (training wheels for balance).
                roll, pitch, _ = _quat_to_euler(d.qpos[3:7])
                # angular velocity of pelvis body:
                wx, wy = d.qvel[3], d.qvel[4]
                mx = eff_harness * (-HARNESS_TILT_KP * roll - HARNESS_TILT_KD * wx)
                my = eff_harness * (-HARNESS_TILT_KP * pitch - HARNESS_TILT_KD * wy)
                d.xfrc_applied[self._pelvis, 3] = float(
                    np.clip(mx, -HARNESS_MMAX, HARNESS_MMAX))
                d.xfrc_applied[self._pelvis, 4] = float(
                    np.clip(my, -HARNESS_MMAX, HARNESS_MMAX))
            mujoco.mj_step(self.model, d)
        harness_f = float(d.xfrc_applied[self._pelvis, 2])
        harness_m = float(abs(d.xfrc_applied[self._pelvis, 3])
                          + abs(d.xfrc_applied[self._pelvis, 4]))
        d.xfrc_applied[self._pelvis, :] = 0.0
        self._steps += 1

        pelvis_pos = d.xpos[self._pelvis]
        vx, vy, vz = d.qvel[0:3]
        roll, pitch, _ = _quat_to_euler(d.qpos[3:7])
        torques = d.qfrc_actuator[self._dof_adr]
        lfoot_z = float(d.xpos[self._lfoot][2])
        rfoot_z = float(d.xpos[self._rfoot][2])
        lfoot_x = float(d.xpos[self._lfoot][0])
        rfoot_x = float(d.xpos[self._rfoot][0])
        min_foot_z = min(lfoot_z, rfoot_z)
        min_knee_z = float(min(d.xpos[self._lknee][2], d.xpos[self._rknee][2]))

        # --- v3 shaped reward (all eight ideas) ---
        r_band = W_BAND * _band_vx(float(vx))
        r_dodge = -(W_VY * abs(float(vy)) + W_Y * abs(float(pelvis_pos[1])))
        energy = W_ENERGY * float(np.sum(torques ** 2))
        # Torso posture: upright torso + slight forward lean (exp kernel,
        # max at roll=0, pitch=PITCH_TGT). Replaces the old tilt penalty.
        r_posture = W_POSTURE * float(np.exp(
            -POSTURE_K * (roll ** 2 + (pitch - PITCH_TGT) ** 2)))

        # Reference-gait tracking: imitation prior for the walking pattern.
        r_track = 0.0
        if self.track_w > 0.0:
            self._phase += 2.0 * np.pi * GAIT_FREQ * (N_SUBSTEPS * PHYS_DT)
            q_leg = d.qpos[7:][LEG_JNTS]
            mse = float(np.mean((q_leg - self._gait_ref()) ** 2))
            r_track = self.track_w * float(np.exp(-TRACK_K * mse))

        r_clear, r_knee = 0.0, 0.0
        if self.n_stairs > 0 and (X0 - 0.6) <= pelvis_pos[0] <= (
            X0 + self.n_stairs * STEP_D + 0.3
        ):
            r_clear = W_CLEAR * float(np.clip((min_foot_z - 0.02) / 0.12, 0.0, 1.0))
            r_knee = W_KNEE * float(np.clip((min_knee_z - 0.45) / 0.20, 0.0, 1.0))

        # Idea 4: split level bonus — half when the first foot crosses a
        # level, half when the second foot joins it (no straddling).
        r_level = 0.0
        for k in range(1, self.n_stairs + 1):
            crossed = self._levels.get(k, set())
            if crossed != {"l", "r"}:
                stair_x0 = X0 + (k - 1) * STEP_D
                stair_x1 = X0 + k * STEP_D
                for foot, fz, fx in (("l", lfoot_z, lfoot_x),
                                     ("r", rfoot_z, rfoot_x)):
                    if (foot not in crossed and fz > k * self.step_h
                            and stair_x0 - 0.05 <= fx <= stair_x1 + 0.05):
                        crossed.add(foot)
                        r_level += self.r_level * 0.5
                self._levels[k] = crossed

        # Ideas 2/6/7: per-foot touchdown events.
        r_alt, r_place, r_soft = 0.0, 0.0, 0.0
        dt = N_SUBSTEPS * PHYS_DT
        lfoot_y = float(d.xpos[self._lfoot][1])
        rfoot_y = float(d.xpos[self._rfoot][1])
        for foot, fz, fx, fy, prev_fz in (
            ("l", lfoot_z, lfoot_x, lfoot_y, self._prev_lfoot_z),
            ("r", rfoot_z, rfoot_x, rfoot_y, self._prev_rfoot_z),
        ):
            if prev_fz > STRIKE_HIGH and fz <= STRIKE_LOW:
                # Idea 7: landing softness — penalize impact velocity.
                impact_v = max(0.0, (prev_fz - fz) / dt)
                r_soft -= W_IMPACT * min(impact_v, IMPACT_VMAX)
                # Idea 2: alternating cadence, only with forward progress
                # (no stomping one foot in place).
                if (
                    self._last_strike_foot is not None
                    and foot != self._last_strike_foot
                    and self._alt_strikes < MAX_ALT_STRIKES
                    and float(pelvis_pos[0]) - self._last_strike_x > MIN_STRIKE_DX
                ):
                    self._alt_strikes += 1
                    r_alt += R_ALT
                self._last_strike_foot = foot
                self._last_strike_x = float(pelvis_pos[0])
                # Idea 6: foot placement near the next step's center.
                nk = self._next_level()
                if (
                    self.n_stairs > 0
                    and nk <= self.n_stairs
                    and (X0 - 0.6) <= fx <= (X0 + self.n_stairs * STEP_D + 0.3)
                ):
                    tx = X0 + (nk - 1) * STEP_D + STEP_D / 2.0
                    err2 = (fx - tx) ** 2 + fy ** 2
                    r_place += W_PLACE * float(np.exp(-PLACE_K * err2))
        self._prev_lfoot_z = lfoot_z
        self._prev_rfoot_z = rfoot_z

        # Idea 3: single-support balance — pelvis over the stance foot.
        r_bal = 0.0
        l_contact = lfoot_z <= CONTACT_Z
        r_contact = rfoot_z <= CONTACT_Z
        if l_contact != r_contact:  # exactly one foot on the ground
            sx = lfoot_x if l_contact else rfoot_x
            sy = lfoot_y if l_contact else rfoot_y
            dist2 = ((float(pelvis_pos[0]) - sx) ** 2
                     + (float(pelvis_pos[1]) - sy) ** 2)
            r_bal = W_BAL * float(np.exp(-BAL_K * dist2))

        # Idea 8: pelvis height tracking (replaces the bouncy r_up) — the
        # target rises step_h for each fully cleared level.
        n_full = sum(1 for v in self._levels.values() if set(v) == {"l", "r"})
        hz_tgt = self._base_pelvis_z + n_full * self.step_h
        r_h = W_HZ * float(np.exp(-HZ_K * (float(pelvis_pos[2]) - hz_tgt) ** 2))

        # Idea 1: potential-based dense progress toward the next step.
        phi = self._potential(float(pelvis_pos[0]), max(lfoot_z, rfoot_z))
        r_pot = POT_GAMMA * phi - self._prev_phi
        self._prev_phi = phi

        reward = (
            r_band + r_dodge + W_ALIVE - energy + r_posture
            + r_clear + r_knee + r_level + r_track
            + r_pot + r_bal + r_alt + r_place + r_soft + r_h
            - HARNESS_W * harness_f - HARNESS_MW * harness_m
        )

        fallen = (pelvis_pos[2] < FALL_Z) or (abs(roll) > TILT_LIM) or (abs(pitch) > TILT_LIM)
        success = (pelvis_pos[0] > self._success_x) and (
            pelvis_pos[2] > self._success_z
        )

        # Anti-stand: no forward progress for a while -> truncate (not a fall,
        # so no penalty; just stop wasting the episode standing/marching).
        stalled = False
        if self.anti_stand and not fallen:
            if pelvis_pos[0] > self._best_x + ANTI_STAND_DX:
                self._best_x = float(pelvis_pos[0])
                self._last_prog_step = self._steps
            elif self._steps - self._last_prog_step > ANTI_STAND_STEPS:
                stalled = True

        terminated, info = False, {"is_success": bool(success)}
        if fallen:
            reward += R_FALL
            terminated = True
        elif success:
            reward += R_SUCCESS
            terminated = True
        truncated = (
            ((self._steps >= MAX_EPISODE_STEPS) or stalled) and not terminated
        )

        info.update(
            dict(
                r_band=r_band, r_h=r_h, r_clear=r_clear, r_knee=r_knee,
                r_level=r_level, r_alt=r_alt, r_place=r_place, r_soft=r_soft,
                r_bal=r_bal, r_pot=r_pot, r_dodge=r_dodge,
                energy=energy, r_posture=r_posture,
                levels=sum(1 for v in self._levels.values() if v),
                levels_full=n_full, alt_strikes=self._alt_strikes,
                pelvis_x=float(pelvis_pos[0]), pelvis_z=float(pelvis_pos[2]),
                min_foot_z=min_foot_z, min_knee_z=min_knee_z,
                roll=float(roll), pitch=float(pitch),
            )
        )
        if not np.isfinite(reward):
            raise RuntimeError(f"non-finite reward: {reward} info={info}")
        return self._get_obs(), float(reward), terminated, truncated, info
