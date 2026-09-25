"""Balls of solitude, MuJoCo edition.

Our farewell to Isaac Gym: 1080 free-falling colored balls in MuJoCo.
Run:  python balls_of_solitude.py   (a viewer window will open)
"""
import numpy as np
import mujoco
import mujoco.viewer

N = 1080
rng = np.random.default_rng(0)
colors = rng.random((N, 3)) * 0.7 + 0.2

bodies = []
for i in range(N):
    x, y = rng.uniform(-4, 4, 2)
    z = rng.uniform(1, 6)
    r, g, b = colors[i]
    bodies.append(
        f'<body pos="{x:.2f} {y:.2f} {z:.2f}">'
        "<freejoint/>"
        f'<geom type="sphere" size="0.15" rgba="{r:.2f} {g:.2f} {b:.2f} 1"/>'
        "</body>"
    )

xml = f"""
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light diffuse=".6 .6 .6" pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="6 6 0.1" rgba="0.9 0.9 0.9 1"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.mjData(model)
print(f"Simulating {N} balls. Close the viewer window to exit.")
mujoco.viewer.launch(model, data)
