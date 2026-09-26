"""Render an evaluation rollout of a trained v1 (g1_stairs_env_v2) policy to MP4.

Mirrors render_eval.py but builds G1StairsEnvV2 with configurable stairs /
harness so the video matches the trained regime (e.g. H14: 6x0.12m stairs,
harness=0.2, r_level=5.0).

Headless rendering: on Linux without a display the script selects
MUJOCO_GL=osmesa (needs libosmesa6 installed).

Example:
    python scripts/render_eval_v2.py --model runs/night_h14_full/final_model.zip \\
        --out runs/night_h14_full/eval_video.mp4 --episodes 3
"""

import argparse
import os
import re
import sys
from pathlib import Path

if os.environ.get("MUJOCO_GL") is None:
    if os.name == "nt" or os.environ.get("DISPLAY"):
        pass
    else:
        os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np

# Windows torch>=2.14 workaround: torch's C++ zip reader fails with a miniz
# error when reading SB3 checkpoints through ZipExtFile streams, even though
# the files are intact. Buffering entries through BytesIO sidesteps it.
# Harmless on other platforms (just an in-memory copy).
import io as _io
import zipfile as _zipfile
_orig_zip_open = _zipfile.ZipFile.open


def _buffered_zip_open(self, name, mode="r", *args, **kwargs):
    f = _orig_zip_open(self, name, mode, *args, **kwargs)
    if mode == "r":
        data = f.read()
        f.close()
        return _io.BytesIO(data)
    return f


_zipfile.ZipFile.open = _buffered_zip_open


def _try_init_renderer(model):
    import mujoco
    try:
        r = mujoco.Renderer(model, height=480, width=640)
        print(f"[render] renderer OK (MUJOCO_GL={os.environ.get('MUJOCO_GL', 'default')})")
        return r
    except Exception as e:  # noqa: BLE001
        print(f"[render] renderer failed: {type(e).__name__}: {e}")
        return None


def _make_follow_camera(model):
    import mujoco
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    root_id = 1
    for name in ("pelvis", "base", "root", "torso"):
        try:
            root_id = model.body(name).id
            break
        except Exception:
            continue
    cam.trackbodyid = root_id
    cam.distance = 3.0
    cam.azimuth = 135.0
    cam.elevation = -12.0
    return cam


def parse_args():
    p = argparse.ArgumentParser(description="Render v1 stairs policy rollout to MP4")
    p.add_argument("--model", type=str, required=True,
                   help="path to PPO checkpoint (.zip suffix optional)")
    p.add_argument("--vecnormalize", type=str, default=None,
                   help="path to vecnormalize .pkl (default: guessed next to --model)")
    p.add_argument("--out", type=str, default="eval_v2.mp4")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--n-stairs", type=int, default=6)
    p.add_argument("--step-h", type=float, default=0.12)
    p.add_argument("--harness", type=float, default=0.2)
    p.add_argument("--r-level", type=float, default=5.0)
    p.add_argument("--static-camera", action="store_true",
                   help="disable the pelvis-tracking camera (fixed view)")
    p.add_argument("--env", type=str, default="v1", choices=["v1", "v3"],
                   help="reward/env version the checkpoint was trained with")
    return p.parse_args()


def main():
    args = parse_args()
    orig_cwd = Path.cwd()

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (orig_cwd / p)

    os.chdir(Path(__file__).resolve().parent)

    model_path = _resolve(args.model)
    if model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")
    if not model_path.exists():
        sys.exit(f"[render] checkpoint not found: {model_path}")
    load_stem = model_path.with_suffix("")  # PPO.load appends .zip itself
    vn_path = _resolve(args.vecnormalize) if args.vecnormalize else None
    if vn_path is None:
        cands = list(model_path.parent.glob("*vecnormalize*.pkl"))
        # Prefer the stats file whose step number matches the checkpoint.
        m = re.search(r"(\d+)_steps$", model_path.stem)
        if m:
            same = [c for c in cands if m.group(1) in c.name]
            if same:
                cands = same
        vn_path = cands[0] if cands else None
    if vn_path is None or not vn_path.exists():
        sys.exit(f"[render] VecNormalize stats not found near {model_path}. "
                 "Pass --vecnormalize explicitly.")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    if args.env == "v3":
        from g1_stairs_env_v3 import G1StairsEnvV3 as EnvCls
    else:
        from g1_stairs_env_v2 import G1StairsEnvV1 as EnvCls

    venv = DummyVecEnv([lambda: EnvCls(
        n_stairs=args.n_stairs, step_h=args.step_h,
        harness=args.harness, r_level=args.r_level)])
    venv = VecNormalize.load(str(vn_path), venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(str(load_stem), env=venv)

    renderer = _try_init_renderer(venv.envs[0].model)
    if renderer is None:
        sys.exit(1)
    follow_cam = None
    if not args.static_camera:
        follow_cam = _make_follow_camera(venv.envs[0].model)
        print(f"[render] pelvis-tracking camera on (body {follow_cam.trackbodyid})")

    import imageio
    frames = []
    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = venv.step(action)
            ep_rew += float(reward[0])
            renderer.update_scene(venv.envs[0].data, camera=follow_cam)
            frames.append(renderer.render())
        success = bool(infos[0].get("is_success", False))
        print(f"[render] episode {ep}: reward={ep_rew:.1f} success={success} "
              f"frames={len(frames)}")
    renderer.close()

    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stride = max(100 // args.fps, 1)
    imageio.mimsave(str(out), frames[::stride], fps=args.fps,
                    codec="libx264", quality=8)
    print(f"[render] wrote {out} ({len(frames[::stride])} frames)")


if __name__ == "__main__":
    main()
