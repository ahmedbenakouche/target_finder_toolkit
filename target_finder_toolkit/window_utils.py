"""Window helpers shared by Qt overlays."""

from __future__ import annotations

import sys
from ctypes import c_void_p


def raise_macos_window_above_system_ui(widget, *, level_offset: int = 0) -> bool:
    """Raise a Qt widget above the macOS menu bar/Dock without using Spaces fullscreen."""
    if sys.platform != "darwin":
        return False
    try:
        import AppKit
        import objc

        widget.winId()
        native_obj = objc.objc_object(c_void_p=c_void_p(int(widget.winId())))
        ns_window = native_obj.window() if hasattr(native_obj, "window") else native_obj
        if ns_window is None or not hasattr(ns_window, "setLevel_"):
            return False

        level = int(AppKit.NSStatusWindowLevel) + int(level_offset)
        ns_window.setLevel_(level)
        behavior = (
            int(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces)
            | int(AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary)
        )
        if hasattr(ns_window, "setCollectionBehavior_"):
            ns_window.setCollectionBehavior_(behavior)
        return True
    except Exception:
        return False
