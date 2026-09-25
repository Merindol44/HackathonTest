"""G1 stair-climbing env v1: curriculum-capable geometry + shaped rewards.

Same 73-dim proprioceptive observation and 29-dim action space as
scripts/g1_stairs_env.py (kept untouched), but:

* Geometry is parameterized: ``n_stairs`` (0 = flat ground) and ``step_h``,
  so one class serves every curriculum stage.
* Reward v1 ("band+clearance") replaces the raw forward-velocity term that
  caused the dense-reward locomotion trap (sprint forward, fall at stairs):
    - trapezoidal forward-velocity band: full reward for vx in [0.25, 1.0]
      m/s, ramping to 0 outside (no more sprint-harvesting);
    - foot-clearance bonus near the stairs (min ankle height);
    - knee-lift bonus near the stairs (min knee-link height);
    - upward-only vz reward (no reward for falling down fast);
    - +2.0 per NEW stair level reached (discrete step-count bonus);
    - +0.15 per foot-strike (cadence), capped at 30/episode;
    - anti side-dodge: -0.4*|vy| - 0.2*|y|;
    - alive +0.10, energy and tilt penalties, fall -10 / success +50
      terminations as in v0.

Run from scripts/:  python -c "from g1_stairs_env_v2 import G1StairsEnvV1; ..."
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
    W_TILT,
    X0,
    G1StairsEnv,
    _quat_to_euler,
)

import mujoco
from gymnasium import spaces

# v1 reward weights
W_BAND = 1.5      # forward-velocity band
W_UP = 2.0        # upward-only vz
W_CLEAR = 1.0     # foot clearance near stairs
W_KNEE = 0.5      # knee lift near stairs
R_LEVEL = 2.0     # per new stair level reached
R_STRIKE = 0.15   # per foot strike
MAX_STRIKES = 30
W_VY = 0.4        # anti side-dodge
W_Y = 0.2

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


class G1StairsEnvV1(G1StairsEnv):
    """Curriculum-capable stairs env with v1 shaped rewards.

    Args:
        n_stairs: number of steps (0 = flat ground, walking task).
        step_h: height of each step (m).
    """

    def __init__(self, g1_xml=None, n_stairs=6, step_h=0.12, render_mode=None):
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
        self._levels = set()
        self._strikes = 0
        self._prev_min_foot_z = 0.0

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Baseline pelvis height for level thresholds (measured, not assumed).
        self._base_pelvis_z = float(self.data.xpos[self._pelvis][2])
        self._levels = set()
        self._strikes = 0
        self._prev_min_foot_z = float(
            min(self.data.xpos[self._lfoot][2], self.data.xpos[self._rfoot][2])
        )
        if self.n_stairs > 0:
            self._success_z = self._base_pelvis_z + self.n_stairs * self.step_h - 0.15
        else:
            self._success_z = 0.5
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self._stand_act + action * (ACTION_SCALE * self._jnt_span)
        target = np.clip(target, self._jnt_lo, self._jnt_hi)
        d = self.data
        for _ in range(N_SUBSTEPS):
            d.ctrl[:] = target
            mujoco.mj_step(self.model, d)
        self._steps += 1

        pelvis_pos = d.xpos[self._pelvis]
        vx, vy, vz = d.qvel[0:3]
        roll, pitch, _ = _quat_to_euler(d.qpos[3:7])
        torques = d.qfrc_actuator[self._dof_adr]
        lfoot_z = float(d.xpos[self._lfoot][2])
        rfoot_z = float(d.xpos[self._rfoot][2])
        min_foot_z = min(lfoot_z, rfoot_z)
        min_knee_z = float(min(d.xpos[self._lknee][2], d.xpos[self._rknee][2]))

        # --- v1 shaped reward ---
        r_band = W_BAND * _band_vx(float(vx))
        r_up = W_UP * float(np.clip(vz, 0.0, 1.0))
        r_dodge = -(W_VY * abs(float(vy)) + W_Y * abs(float(pelvis_pos[1])))
        energy = W_ENERGY * float(np.sum(torques ** 2))
        tilt = W_TILT * float(roll ** 2 + pitch ** 2)

        r_clear, r_knee = 0.0, 0.0
        if self.n_stairs > 0 and (X0 - 0.6) <= pelvis_pos[0] <= (
            X0 + self.n_stairs * STEP_D + 0.3
        ):
            r_clear = W_CLEAR * float(np.clip((min_foot_z - 0.02) / 0.12, 0.0, 1.0))
            r_knee = W_KNEE * float(np.clip((min_knee_z - 0.45) / 0.20, 0.0, 1.0))

        # Discrete level bonus: pelvis rises a half step over each step.
        r_level = 0.0
        for k in range(1, self.n_stairs + 1):
            if k not in self._levels and pelvis_pos[2] > self._base_pelvis_z + (
                k - 0.5
            ) * self.step_h and pelvis_pos[0] > X0 + (k - 1) * STEP_D - 0.05:
                self._levels.add(k)
                r_level += R_LEVEL

        # Foot-strike cadence: foot was up, now it lands.
        r_strike = 0.0
        if (
            self._prev_min_foot_z > STRIKE_HIGH
            and min_foot_z <= STRIKE_LOW
            and self._strikes < MAX_STRIKES
        ):
            self._strikes += 1
            r_strike = R_STRIKE
        self._prev_min_foot_z = min_foot_z

        reward = (
            r_band + r_up + r_dodge + W_ALIVE - energy - tilt
            + r_clear + r_knee + r_level + r_strike
        )

        fallen = (pelvis_pos[2] < FALL_Z) or (abs(roll) > TILT_LIM) or (abs(pitch) > TILT_LIM)
        success = (pelvis_pos[0] > self._success_x) and (
            pelvis_pos[2] > self._success_z
        )

        terminated, info = False, {"is_success": bool(success)}
        if fallen:
            reward += R_FALL
            terminated = True
        elif success:
            reward += R_SUCCESS
            terminated = True
        truncated = (self._steps >= MAX_EPISODE_STEPS) and not terminated

        info.update(
            dict(
                r_band=r_band, r_up=r_up, r_clear=r_clear, r_knee=r_knee,
                r_level=r_level, r_strike=r_strike, r_dodge=r_dodge,
                energy=energy, tilt=tilt,
                levels=len(self._levels), strikes=self._strikes,
                pelvis_x=float(pelvis_pos[0]), pelvis_z=float(pelvis_pos[2]),
                min_foot_z=min_foot_z, min_knee_z=min_knee_z,
                roll=float(roll), pitch=float(pitch),
            )
        )
        if not np.isfinite(reward):
            raise RuntimeError(f"non-finite reward: {reward} info={info}")
        return self._get_obs(), float(reward), terminated, truncated, info
