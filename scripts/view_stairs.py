"""Interactive viewer for the G1 stair-climbing environment.

Run from the repo root:
    python scripts/view_stairs.py

Opens a MuJoCo viewer window showing the G1 standing before the staircase
(zero action = hold standing posture). Drag to rotate, scroll to zoom,
right-drag to pan. Close the window to exit.

Requires the G1 model — fetch it first with:
    python scripts/download_g1_model.py
"""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_stairs_env import G1StairsEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
G1_XML = REPO_ROOT / "assets" / "mujoco_menagerie" / "unitree_g1" / "g1.xml"


def main():
    if not G1_XML.exists():
        print(f"G1 model not found at {G1_XML}")
        print("Fetch it first:  python scripts/download_g1_model.py")
        sys.exit(1)

    env = G1StairsEnv()
    env.reset(seed=0)
    print("Viewer open: G1 standing before the 6-step staircase.")
    print("Close the window to exit.")
    zero = np.zeros(29, dtype=np.float32)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            env.step(zero)
            viewer.sync()
            time.sleep(0.01)  # ~real time (100 Hz control)
    env.close()


if __name__ == "__main__":
    main()
