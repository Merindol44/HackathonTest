"""Headless verification: load the Unitree G1 MJCF model and step the sim.

Run with the project venv:
    ~/workspace/hackathon/.venv/bin/python ~/workspace/hackathon/scripts/verify_g1_loads.py
"""
import os
import time

import mujoco
import numpy as np

MODEL_PATH = os.path.expanduser(
    "~/workspace/hackathon/assets/mujoco_menagerie/unitree_g1/g1.xml"
)
N_STEPS = 100


def main() -> None:
    assert os.path.exists(MODEL_PATH), f"model not found: {MODEL_PATH}"

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print(f"model      : {MODEL_PATH}")
    print(f"bodies     : {model.nbody}")
    print(f"joints     : {model.njnt}")
    print(f"dof (nv)   : {model.nv}")
    print(f"actuators  : {model.nu}")
    print(f"timestep   : {model.opt.timestep}s")

    # Random actions within actuator control ranges, then zero actions.
    rng = np.random.default_rng(0)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    span = hi - lo

    t0 = time.time()
    for i in range(N_STEPS):
        if i < N_STEPS // 2:
            data.ctrl[:] = lo + span * rng.random(model.nu)
        else:
            data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        if data.ncon < 0:
            raise RuntimeError("invalid contact count")
    wall = time.time() - t0

    qpos_finite = bool(np.all(np.isfinite(data.qpos)))
    print(f"steps      : {N_STEPS} ok ({wall:.2f}s wall, "
          f"{N_STEPS / wall:.0f} steps/s)")
    print(f"qpos finite: {qpos_finite}")
    print(f"sim time   : {data.time:.3f}s")
    assert qpos_finite, "NaN/Inf in qpos after stepping"
    print("VERIFY OK: G1 model loads and steps headless without errors.")


if __name__ == "__main__":
    main()
