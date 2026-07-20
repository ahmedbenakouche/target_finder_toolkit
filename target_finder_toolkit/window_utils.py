"""Window helpers shared by Qt overlays."""

from __future__ import annotations

import sys
import traceback
from ctypes import c_void_p

_qt_crash_guard_installed = False


def install_qt_crash_guard() -> None:
    """Stop PyQt6 from calling ``abort()`` on an unhandled slot exception.

    PyQt6's default ``sys.excepthook`` is the unmodified interpreter one; when
    a Python exception escapes a slot invoked from C++ (e.g. a QTimer
    callback such as the gaze/paint update loop), PyQt6 prints the traceback
    and then calls ``qFatal()``, which aborts the whole process
    (EXC_CRASH/SIGABRT). On macOS that abort tears down the app's fullscreen
    window/Space immediately, which is what shows up as "jumps straight to
    the desktop" right when something goes wrong mid-session. Installing any
    custom ``sys.excepthook`` disables that abort-on-exception behavior, so a
    bug in a single timer tick is logged and the app keeps running instead of
    crashing outright.
    """
    global _qt_crash_guard_installed
    if _qt_crash_guard_installed:
        return
    _qt_crash_guard_installed = True

    def _hook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def warm_up_macos_keyboard_layout() -> None:
    """Pre-fetch the current keyboard input source on the main thread.

    pynput's macOS keyboard.Listener resolves the active keyboard layout
    (TISCopyCurrentKeyboardInputSource / TISGetInputSourceProperty) the first
    time its background listener thread starts. On recent macOS versions that
    lazy lookup happens off the main thread and can trip an internal
    ``dispatch_assert_queue`` check, crashing the whole process with
    EXC_BREAKPOINT/SIGTRAP (seen right after eye calibration, when the
    keyboard listener thread handles its first real key event). Calling the
    same lookup once from the main thread before starting the listener
    populates Apple's cache so the later off-thread call is a cheap hit
    instead of a cold, assert-checked one.
    """
    if sys.platform != "darwin":
        return
    try:
        from pynput._util.darwin import keycode_context

        with keycode_context():
            pass
    except Exception:
        pass


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
