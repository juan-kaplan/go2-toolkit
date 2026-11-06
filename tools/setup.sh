#!/usr/bin/env bash
# One time setup: vendored sources into third_party/, then the environment.
#
#     bash tools/setup.sh
#
# There is nothing to activate afterwards. uv owns .venv/ and `uv run` picks it
# up, so every command in the README works from a plain shell.
#
# This used to build CycloneDDS from source with cmake, which is why it was
# called setup_dds.sh. That was unnecessary: cyclonedds ships a manylinux wheel
# that bundles its own libddsc, and the locally built copy was never the one
# Python loaded. DDS itself is untouched and still carries every LowCmd and
# LowState; only the build step is gone. If third_party/cyclonedds exists from
# an older checkout it is dead weight and safe to delete.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$REPO_ROOT/third_party"
mkdir -p "$THIRD_PARTY"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with one of:" >&2
  echo "    pip install uv" >&2
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo ">> cloning vendored sources"
# The SDK is not on PyPI and is installed from here as an editable dependency,
# declared in pyproject.toml under [tool.uv.sources].
[[ -d "$THIRD_PARTY/unitree_sdk2_python" ]] || git clone --depth 1 \
  https://github.com/unitreerobotics/unitree_sdk2_python.git \
  "$THIRD_PARTY/unitree_sdk2_python"
# The simulator: the Go2 MuJoCo model plus the bridge that puts it behind the
# real DDS topics. Not a Python dependency, so it is cloned and not installed.
[[ -d "$THIRD_PARTY/unitree_mujoco" ]] || git clone --depth 1 \
  https://github.com/unitreerobotics/unitree_mujoco.git \
  "$THIRD_PARTY/unitree_mujoco"

# Remove hurdle and staircase obstacles so the Go2 walks on an infinite flat floor
python3 -c "
import xml.etree.ElementTree as ET
scene_path = '$THIRD_PARTY/unitree_mujoco/unitree_robots/go2/scene.xml'
tree = ET.parse(scene_path)
root = tree.getroot()
worldbody = root.find('worldbody')
if worldbody is not None:
    for geom in list(worldbody.findall('geom')):
        if geom.get('type') == 'box':
            worldbody.remove(geom)
    tree.write(scene_path)
"

echo ">> resolving and installing the environment"
cd "$REPO_ROOT"
uv sync --extra plots

echo ">> verifying"
uv run python -c "
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from cyclonedds.domain import DomainParticipant
import mujoco, numpy, scipy, sys
assert not any('.local/lib' in p for p in sys.path), 'user site leaked into the venv'
print('  DDS, SDK, MuJoCo all import; environment is isolated')
"

cat <<'EOF'

done. Nothing to activate. Next:

    uv run python tools/run_bridge.py --viewer &
    uv run python examples/run_gait.py --gait walk

Optional, for plots and trial videos:

    uv sync --extra plots
EOF
