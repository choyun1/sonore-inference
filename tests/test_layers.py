"""The differentiable core must not depend on sonore (design D2).

sonore is allowed only in the modules that make stimuli or plots. Every other
module is imported in a fresh interpreter, and the test fails if that pulls
sonore in.
"""

import pkgutil
import subprocess
import sys

import sonore_inference

# Modules allowed to import sonore, by dotted name prefix.
SONORE_ALLOWED = ("sonore_inference.stimuli",)


def _core_modules() -> list[str]:
    names = [sonore_inference.__name__]
    for module in pkgutil.walk_packages(sonore_inference.__path__, prefix="sonore_inference."):
        if not module.name.startswith(SONORE_ALLOWED):
            names.append(module.name)
    return names


def test_core_does_not_import_sonore():
    for name in _core_modules():
        code = f"import sys, {name}; sys.exit('sonore' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code])
        assert result.returncode == 0, f"{name} imports sonore"
