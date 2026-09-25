#!/bin/bash
# Download the Unitree G1 MuJoCo model (Menagerie, sparse checkout — unitree_g1 only).
# Run from the repo root:  bash scripts/download_g1_model.sh
# Result: assets/mujoco_menagerie/unitree_g1/g1.xml (+ meshes)
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$REPO_ROOT/assets/mujoco_menagerie"

if [ -f "$TARGET/unitree_g1/g1.xml" ]; then
  echo "G1 model already present at $TARGET/unitree_g1/g1.xml"
  exit 0
fi

mkdir -p "$REPO_ROOT/assets"
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$TARGET"
cd "$TARGET"
git sparse-checkout set unitree_g1
echo "Done: $TARGET/unitree_g1/g1.xml"
