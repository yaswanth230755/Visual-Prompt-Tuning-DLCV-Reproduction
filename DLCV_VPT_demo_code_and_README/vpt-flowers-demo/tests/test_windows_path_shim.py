"""Windows-path regression test (no weights needed): python tests/test_windows_path_shim.py

On Windows `os.path` is `ntpath`, whose join inserts a backslash; the official vit.py binds `from os.path import join as pjoin`
and builds the keys of the .npz checkpoint with it, so loading failed with KeyError (npz keys use '/').
official_runner.prepare() imports that module with posixpath.join bound instead, WITHOUT editing any official file.
We simulate Windows by rebinding `os.path` to `ntpath` only while the official module is (re)imported.
"""
import ntpath
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import official_runner as R  # noqa: E402

R.prepare(str(ROOT / "third_party" / "vpt_official"))       # normal setup (imports the official module once)
MOD = "src.models.vit_backbones.vit"
KEY_PARTS = ("Transformer/encoderblock_0", "MultiHeadDotProductAttention_1/query", "kernel")
GOOD = "Transformer/encoderblock_0/MultiHeadDotProductAttention_1/query/kernel"


def import_under_windows(fix: bool):
    """Re-import the official module as it would be imported on Windows; return its `pjoin`."""
    sys.modules.pop(MOD, None)
    real_path, real_modpath, real_join = os.path, sys.modules["os.path"], ntpath.join
    os.path = sys.modules["os.path"] = ntpath               # simulate Windows (`from os.path import ...` reads sys.modules)
    try:
        if fix:
            R._import_vit_with_posix_join()
        else:
            import importlib
            importlib.import_module(MOD)
    finally:
        os.path, sys.modules["os.path"] = real_path, real_modpath
        assert ntpath.join is real_join, "ntpath.join must be restored"
    return sys.modules[MOD].pjoin


bug = import_under_windows(fix=False)(*KEY_PARTS)
assert bug != GOOD and "\\" in bug, f"expected the Windows bug to reproduce, got {bug!r}"
print(f"[info] without the shim (Windows): {bug!r}  -> KeyError when reading the checkpoint")
fixed = import_under_windows(fix=True)(*KEY_PARTS)
assert fixed == GOOD, fixed
print("[PASS] Windows path shim: checkpoint keys use '/', os.path.join is restored, no official file edited")
