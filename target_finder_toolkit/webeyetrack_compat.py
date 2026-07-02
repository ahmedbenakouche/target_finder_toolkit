from __future__ import annotations

import sys


def patch_webeyetrack_dataclass_defaults() -> None:
    """Allow older WebEyeTrack dataclasses with numpy array defaults on Windows.

    Some WebEyeTrack releases define dataclass fields with numpy.ndarray
    defaults. Recent Python/dataclasses versions reject these mutable defaults
    during import. Patch dataclasses only for that specific case before
    importing WebEyeTrack.
    """
    if not sys.platform.startswith("win"):
        return

    try:
        import dataclasses
        import numpy as np
    except Exception:
        return

    if getattr(dataclasses, "_target_finder_webeyetrack_patch", False):
        return

    original_get_field = getattr(dataclasses, "_get_field", None)
    if not callable(original_get_field):
        return

    def patched_get_field(cls, a_name, a_type, default_kw_only):
        try:
            return original_get_field(cls, a_name, a_type, default_kw_only)
        except ValueError as exc:
            message = str(exc)
            if "mutable default" not in message or "numpy.ndarray" not in message:
                raise
            default = getattr(cls, a_name, dataclasses.MISSING)
            if not isinstance(default, np.ndarray):
                raise
            value = default.copy()
            setattr(
                cls,
                a_name,
                dataclasses.field(default_factory=lambda value=value: value.copy()),
            )
            return original_get_field(cls, a_name, a_type, default_kw_only)

    dataclasses._get_field = patched_get_field
    dataclasses._target_finder_webeyetrack_patch = True
