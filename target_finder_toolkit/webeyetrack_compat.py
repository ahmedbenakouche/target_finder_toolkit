from __future__ import annotations

import copy


def patch_webeyetrack_dataclass_defaults() -> None:
    """Allow older WebEyeTrack dataclasses with mutable defaults to import.

    Some WebEyeTrack releases define dataclass fields with mutable defaults
    such as numpy.ndarray or nested WebEyeTrack config objects. Python 3.11+
    rejects these during import on every platform (dataclasses now considers
    unhashable defaults, including numpy arrays, invalid) -- this used to only
    reproduce on Windows setups that happened to pin a newer Python, but it
    reproduces identically on macOS/Linux with Python 3.11+. Patch dataclasses
    only for WebEyeTrack-related mutable defaults before importing WebEyeTrack.
    """
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

    def is_webeyetrack_mutable_default(value) -> bool:
        if isinstance(value, np.ndarray):
            return True
        value_type = type(value)
        module = getattr(value_type, "__module__", "")
        return module.startswith("webeyetrack")

    def copy_default(value):
        if isinstance(value, np.ndarray):
            return value.copy()
        try:
            return copy.deepcopy(value)
        except Exception:
            return copy.copy(value)

    def patched_get_field(cls, a_name, a_type, default_kw_only):
        try:
            return original_get_field(cls, a_name, a_type, default_kw_only)
        except ValueError as exc:
            message = str(exc)
            if "mutable default" not in message:
                raise
            default = getattr(cls, a_name, dataclasses.MISSING)
            if not is_webeyetrack_mutable_default(default):
                raise
            value = copy_default(default)
            setattr(
                cls,
                a_name,
                dataclasses.field(default_factory=lambda value=value: copy_default(value)),
            )
            return original_get_field(cls, a_name, a_type, default_kw_only)

    dataclasses._get_field = patched_get_field
    dataclasses._target_finder_webeyetrack_patch = True
