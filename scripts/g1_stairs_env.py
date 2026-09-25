"""Gymnasium environment: Unitree G1 humanoid climbing a staircase (MuJoCo).

Task (from the SKF hackathon brief): train a learning-based controller so the
G1 walks up a flight of stairs. The MJCF is composed at runtime: the MuJoCo
Menagerie G1 model plus a staircase (6 steps, 0.12 m high x 0.30 m deep) and a
top platform, built with mujoco.MjSpec so mesh paths keep resolving.

Reward design (all terms documented here):
  * r_forward  = 1.5 * clip(pelvis_vx, -1, 2)        -- walk toward the stairs
  * r_up       = 2.0 * clip(pelvis_vz, -0.5, 1)      -- gain height on steps
  * r_alive    = +0.10 per step                      -- stay alive bonus
  * r_energy   = -2e-6 * sum(actuator_torques^2)     -- penalize thrashing
  * r_tilt     = -1.0 * (roll^2 + pitch^2)           -- stay upright
  * fall       : pelvis_z < 0.35 m or |roll|/|pitch| > 0.8 rad
                 -> reward -= 10, episode terminates
  * success    : pelvis_x > 3.6 m and pelvis_z > 1.30 m (on top platform)
                 -> reward += 50, episode terminates
  * timeout    : 1000 env steps (10 s sim time) -> truncated

Action: 29-dim vector in [-1, 1], mapped to target joint angles as
  target = stand_pose + action * 0.35 * joint_half_range
(clipped to each joint's limits), fed to the model's built-in position
servos (kp=500, kd~43). Zero action therefore means "hold the standing
posture", which makes the task learnable; the 0.35 factor still lets the
policy reach most of each joint's range. Commands are applied at 100 Hz
(5 physics substeps of 0.002 s per env step).

Observation (73-dim, proprioceptive only, no vision):
  * 29 joint positions, normalized to [-1, 1] via joint ranges
  * 29 joint velocities, scaled by 1/10 and clipped
  * 3 pelvis position (x*0.2, y*0.5, z*0.5)
  * 3 pelvis linear velocity / 2, clipped
  * 3 torso euler angles (roll, pitch, yaw) in radians
  * 6 foot positions relative to pelvis / 1.5
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# ----------------------------------------------------------------------------
# Task geometry
# ----------------------------------------------------------------------------
N_STEPS = 6          # number of stair steps
STEP_H = 0.12        # step height (m)
STEP_D = 0.30        # step depth  (m)
STAIR_W = 2.0        # stair width (m)
X0 = 1.2             # x of the front edge of the first step (m)
TOP_H = N_STEPS * STEP_H            # 0.72 m
PLAT_LEN = 1.8                      # top platform length (m)

# ----------------------------------------------------------------------------
# Episode / control constants
# ----------------------------------------------------------------------------
PHYS_DT = 0.002
N_SUBSTEPS = 5                      # -> 100 Hz control
MAX_EPISODE_STEPS = 1000            # 10 s of sim time

FALL_Z = 0.35          # pelvis height below this = fallen
TILT_LIM = 0.8         # |roll| or |pitch| above this (rad) = fallen
SUCCESS_X = X0 + N_STEPS * STEP_D + 0.6   # 3.6 m, over the top platform
SUCCESS_Z = 1.30       # pelvis height proving we are on top of the stairs

W_FWD, W_UP = 1.5, 2.0
W_ALIVE = 0.10
W_ENERGY = 2e-6
W_TILT = 1.0
R_FALL = -10.0
R_SUCCESS = 50.0

N_ACT = 29
OBS_DIM = 29 + 29 + 3 + 3 + 3 + 6  # = 73
ACTION_SCALE = 0.35  # action in [-1,1] -> +-35% of each joint's half-range


def _build_model(g1_xml_path: Path) -> mujoco.MjModel:
    """Compose G1 + ground + staircase + top platform via MjSpec."""
    spec = mujoco.MjSpec.from_file(str(g1_xml_path))
    world = spec.worldbody

    # Flat ground.
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[8.0, 8.0, 0.1],
        rgba=[0.92, 0.92, 0.94, 1.0],
    )
    # Staircase: solid boxes from the ground, step i has top at (i+1)*STEP_H.
    for i in range(N_STEPS):
        h = (i + 1) * STEP_H
        world.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[STEP_D / 2, STAIR_W / 2, h / 2],
            pos=[X0 + STEP_D / 2 + i * STEP_D, 0.0, h / 2],
            rgba=[0.45, 0.47, 0.55, 1.0],
        )
    # Top platform (greenish so it reads as the goal in videos).
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[PLAT_LEN / 2, STAIR_W / 2, TOP_H / 2],
        pos=[X0 + N_STEPS * STEP_D + PLAT_LEN / 2, 0.0, TOP_H / 2],
        rgba=[0.30, 0.55, 0.35, 1.0],
    )
    return spec.compile()


def _quat_to_euler(q):
    """MuJoCo quat [w, x, y, z] -> (roll, pitch, yaw) in radians."""
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class G1StairsEnv(gym.Env):
    """Unitree G1 stair-climbing environment (see module docstring)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, g1_xml=None, render_mode=None):
        super().__init__()
        if g1_xml is None:
            g1_xml = Path(__file__).resolve().parent.parent / "assets" / \
                "mujoco_menagerie" / "unitree_g1" / "g1.xml"
        self.model = _build_model(Path(g1_xml))
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_ACT,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # "stand" keyframe = neutral posture used at reset and as the
        # action-space center.
        _k = mujoco.mjtObj.mjOBJ_KEY
        self._stand_qpos = self.model.key_qpos[
            mujoco.mj_name2id(self.model, _k, "stand")
        ].copy()

        # Actuator i drives joint actuator_trnid[i, 0]; cache qpos/dof addresses
        # and joint ranges for the action mapping.
        jnt = self.model.actuator_trnid[:, 0].astype(int)
        self._qpos_adr = self.model.jnt_qposadr[jnt]
        self._dof_adr = self.model.jnt_dofadr[jnt]
        lo = self.model.jnt_range[jnt, 0]
        hi = self.model.jnt_range[jnt, 1]
        self._jnt_lo = lo
        self._jnt_hi = hi
        self._jnt_mid = (lo + hi) / 2.0
        self._jnt_span = (hi - lo) / 2.0
        # Neutral posture = actuated joints of the "stand" keyframe; actions
        # are offsets around it (see ACTION_SCALE), clipped to joint limits.
        self._stand_act = self._stand_qpos[7:].copy()

        # Body ids used for observations / termination.
        _b = mujoco.mjtObj.mjOBJ_BODY
        self._pelvis = mujoco.mj_name2id(self.model, _b, "pelvis")
        self._lfoot = mujoco.mj_name2id(self.model, _b, "left_ankle_roll_link")
        self._rfoot = mujoco.mj_name2id(self.model, _b, "right_ankle_roll_link")

        self._steps = 0
        self.render_mode = render_mode
        self._renderer = None

    # ------------------------------------------------------------------
    def _get_obs(self):
        d, m = self.data, self.model
        q = d.qpos[self._qpos_adr]
        qn = (q - self._jnt_mid) / np.maximum(self._jnt_span, 1e-6)
        qv = np.clip(d.qvel[self._dof_adr] / 10.0, -1.0, 1.0)

        pelvis_pos = d.xpos[self._pelvis]
        pelvis_vel = d.qvel[0:3]
        roll, pitch, yaw = _quat_to_euler(d.qpos[3:7])

        feet_rel = np.concatenate(
            [d.xpos[self._lfoot] - pelvis_pos, d.xpos[self._rfoot] - pelvis_pos]
        ) / 1.5

        obs = np.concatenate(
            [
                qn,
                qv,
                pelvis_pos * np.array([0.2, 0.5, 0.5]),
                np.clip(pelvis_vel / 2.0, -1.0, 1.0),
                np.array([roll, pitch, yaw]),
                feet_rel,
            ]
        ).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        d = self.data
        mujoco.mj_resetData(self.model, d)

        qpos = self._stand_qpos.copy()
        # Small randomization so the policy does not overfit one exact start.
        qpos[0] += self.np_random.uniform(-0.05, 0.05)   # x
        qpos[1] += self.np_random.uniform(-0.05, 0.05)   # y
        yaw0 = self.np_random.uniform(-0.05, 0.05)
        qpos[3:7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
        qpos[7:] += self.np_random.uniform(-0.05, 0.05, size=N_ACT)
        d.qpos[:] = qpos
        d.qvel[:] = self.np_random.uniform(-0.01, 0.01, size=self.model.nv)
        mujoco.mj_forward(self.model, d)

        self._steps = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # [-1, 1] -> target joint angle: offset around the standing posture,
        # scaled per joint by 35% of its half-range, clipped to joint limits.
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

        r_fwd = W_FWD * float(np.clip(vx, -1.0, 2.0))
        r_up = W_UP * float(np.clip(vz, -0.5, 1.0))
        energy = W_ENERGY * float(np.sum(torques ** 2))
        tilt = W_TILT * float(roll ** 2 + pitch ** 2)
        reward = r_fwd + r_up + W_ALIVE - energy - tilt

        fallen = (pelvis_pos[2] < FALL_Z) or (abs(roll) > TILT_LIM) or (abs(pitch) > TILT_LIM)
        success = (pelvis_pos[0] > SUCCESS_X) and (pelvis_pos[2] > SUCCESS_Z)

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
                r_fwd=r_fwd, r_up=r_up, energy=energy, tilt=tilt,
                pelvis_x=float(pelvis_pos[0]), pelvis_z=float(pelvis_pos[2]),
                roll=float(roll), pitch=float(pitch),
            )
        )
        if not np.isfinite(reward):
            raise RuntimeError(f"non-finite reward: {reward} info={info}")
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
