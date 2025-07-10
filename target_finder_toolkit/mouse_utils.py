# mouse_utils.py

"""
Utilities for managing system cursor visibility and Windows mouse acceleration.
- hide_cursor_everywhere() / restore_default_cursor(): hide/show system cursor (Windows) or Qt override (other OS)
- get_current_mouse_acceleration_settings(), set_mouse_acceleration(), apply_mouse_settings(): disable_mouse_acceleration_temporarily(): helper to disable and restore acceleration
"""


# 1) Cursor hide/show functions
# -----------------------------

import sys
from PyQt6 import QtWidgets, QtGui, QtCore

# Detect platform
IS_WIN = sys.platform.startswith("win")

if IS_WIN:
    import ctypes
    # Windows API handles
    user32 = ctypes.windll.user32
    SPI_SETCURSORS = 0x57

    # Windows system cursor IDs
    # https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setsystemcursor
    OCR_IDS = [32512, 32513, 32514, 32515, 32516, 32640, 32641, 32642, 32643,
               32644, 32645, 32646, 32648, 32649, 32650, 32651]

    # Transparent cursor
    def create_blank_cursor():
        # AND plane = 0xFF (all bits 1) XOR plane = 0x00 (all bits 0)
        and_plane = (ctypes.c_ubyte * 1)(0xFF)
        xor_plane = (ctypes.c_ubyte * 1)(0x00)
        return user32.CreateCursor(
            None,              # hInstance
            0, 0,              # hot spot
            1, 1,              # width × height
            ctypes.byref(and_plane),  # pvANDPlane
            ctypes.byref(xor_plane)   # pvXORPlane
        )

    # Replace all system cursors with the transparent one on widnows
    def hide_cursor_everywhere():
        if IS_WIN:
            blank = create_blank_cursor()
            for idc in OCR_IDS:
                user32.SetSystemCursor(blank, idc)

    # Reload default cursors from Windows settings
    def restore_default_cursors():
        if IS_WIN:
            user32.SystemParametersInfoW(SPI_SETCURSORS, 0, None, 0)

else:
    # For Linux/macOS à tester !!??? .............
    def hide_cursor_everywhere():
        # Hide the cursor using Qt override (Linux/macOS)
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.BlankCursor))

    def restore_default_cursors():
        # Restore the cursor by restoring Qt override
        QtWidgets.QApplication.restoreOverrideCursor()


# 2) Mouse acceleration functions
# ---------------------------------------

if IS_WIN:
    import winreg

    # Constantes Windows
    SPI_SETMOUSE = 0x0004
    SPIF_SENDCHANGE = 0x02

    _ORIGINAL_MOUSE_ACCEL = None


    def disable_mouse_acceleration():
        """ Disable Windows mouse acceleration (set all thresholds to 0),
         and remember the original settings for later restoration """
        global _ORIGINAL_MOUSE_ACCEL
        if _ORIGINAL_MOUSE_ACCEL is None:
            """
            Read the current mouse acceleration settings from the registry
            Windows uses three values:
             - MouseSpeed: base acceleration level (0=no accel, 1=standard, 2=high)
             - MouseThreshold1: distance threshold 1 (below this no accel)
             - MouseThreshold2: distance threshold 2 (above this stronger accel)
            """
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Control Panel\Mouse") as key:
                speed, _ = winreg.QueryValueEx(key, "MouseSpeed")
                th1, _ = winreg.QueryValueEx(key, "MouseThreshold1")
                th2, _ = winreg.QueryValueEx(key, "MouseThreshold2")
                _ORIGINAL_MOUSE_ACCEL = (speed, th1, th2)

                # Write new mouse acceleration settings into the registry
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Control Panel\Mouse", 0, winreg.KEY_SET_VALUE) as key:
                    winreg.SetValueEx(key, "MouseSpeed", 0, winreg.REG_SZ, "0")
                    winreg.SetValueEx(key, "MouseThreshold1", 0, winreg.REG_SZ, "0")
                    winreg.SetValueEx(key, "MouseThreshold2", 0, winreg.REG_SZ, "0")

                # Apply the registry settings immediately to the running system
                params = (ctypes.c_int * 3)(0, 0, 0)
                ctypes.windll.user32.SystemParametersInfoW(SPI_SETMOUSE, 0, ctypes.byref(params), SPIF_SENDCHANGE)


    def restore_mouse_acceleration():
        """Restore the previously saved Windows mouse acceleration settings"""
        global _ORIGINAL_MOUSE_ACCEL
        if _ORIGINAL_MOUSE_ACCEL:
            speed, th1, th2 = _ORIGINAL_MOUSE_ACCEL

            # Write new mouse acceleration settings into the registry
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Control Panel\Mouse", 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "MouseSpeed", 0, winreg.REG_SZ, speed)
                winreg.SetValueEx(key, "MouseThreshold1", 0, winreg.REG_SZ, th1)
                winreg.SetValueEx(key, "MouseThreshold2", 0, winreg.REG_SZ, th2)

            # Apply the registry settings immediately to the running system
            params = (ctypes.c_int * 3)(int(th1), int(th2), int(speed))
            ctypes.windll.user32.SystemParametersInfoW(SPI_SETMOUSE, 0, ctypes.byref(params), SPIF_SENDCHANGE)

            _ORIGINAL_MOUSE_ACCEL = None

else:
    # No-op on non Windows platforms
    def disable_mouse_acceleration():
        pass

    def restore_mouse_acceleration():
        pass
