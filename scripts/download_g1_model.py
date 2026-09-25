"""Fetch the Unitree G1 MuJoCo model (Menagerie, sparse checkout, ~60 MB).

Run from the repo root:
    python scripts/download_g1_model.py

Requires: git on PATH. Result: assets/mujoco_menagerie/unitree_g1/g1.xml
Works on Windows / macOS / Linux (unlike the .sh version, no bash needed).
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "assets" / "mujoco_menagerie"
G1_XML = TARGET / "unitree_g1" / "g1.xml"
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    if G1_XML.exists():
        print(f"Already present: {G1_XML}")
        return
    try:
        run(["git", "--version"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("git not found on PATH. Install Git first (https://git-scm.com/downloads).")
        sys.exit(1)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print("Cloning MuJoCo Menagerie (sparse, unitree_g1 only)...")
    run(["git", "clone", "--filter=blob:none", "--sparse", MENAGERIE_URL, str(TARGET)])
    run(["git", "sparse-checkout", "set", "unitree_g1"], cwd=str(TARGET))
    print(f"Done: {G1_XML}")


if __name__ == "__main__":
    main()
