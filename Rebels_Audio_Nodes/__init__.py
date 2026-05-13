import importlib
import subprocess
import sys
import os

REQUIRED = {
    "pedalboard": "pedalboard",
    "librosa": "librosa",
    "soundfile": "soundfile",
}

def _install(package):
    print(f"[Rebels_Audio_Nodes] Installing missing dependency: {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

for module, pip_name in REQUIRED.items():
    try:
        importlib.import_module(module)
    except ImportError:
        _install(pip_name)

from .rebel_audio_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
