"""
Utilities to manage system cursor visibility and mouse acceleration.

This module exposes a minimal, public helper API used by the two example
interaction techniques (Bubble Cursor and Semantic Pointing):

- `hide_cursor_everywhere()` / `restore_default_cursors()`:
    Hide/show the system cursor globally (platform-specific).
- `disable_mouse_acceleration()` / `restore_mouse_acceleration()`:
    Disable/restore OS mouse acceleration (platform-specific).

Notes
-----
- Windows: uses User32 APIs to replace system cursors and writes registry keys
  to control acceleration.
- Linux (X11): uses XFixes to hide/show the cursor and `xinput` to flip accel profiles
  (libinput/evdev). Wayland is not supported here.
- macOS: currently no-op stubs (could be implemented via Quartz/Cocoa).
"""



# 1) Cursor hide/show functions
# -----------------------------

import sys
from PyQt6 import QtWidgets, QtGui, QtCore

if sys.platform.startswith("win"):
    import ctypes
    # Windows API handles
    user32 = ctypes.windll.user32
    SPI_SETCURSORS = 0x57
    SPIF_SENDCHANGE = 0x02

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
        for idc in OCR_IDS:
            blank = create_blank_cursor()
            user32.SetSystemCursor(blank, idc)

    # Reload default cursors from Windows settings
    def restore_default_cursors():
        user32.SystemParametersInfoW(SPI_SETCURSORS, 0, None, SPIF_SENDCHANGE)

elif sys.platform.startswith("linux"):
    import ctypes

    # X11 and XFixes libraries
    libX11 = ctypes.cdll.LoadLibrary("libX11.so.6")
    libXfixes = ctypes.cdll.LoadLibrary("libXfixes.so.3")

    # argument and types for X11 functions
    libX11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    libX11.XOpenDisplay.restype = ctypes.c_void_p
    libX11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    libX11.XDefaultRootWindow.restype = ctypes.c_ulong
    libX11.XFlush.argtypes = [ctypes.c_void_p]
    libX11.XFlush.restype = ctypes.c_int

    # argument and return types for XFixes extension
    libXfixes.XFixesQueryVersion.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int)]
    libXfixes.XFixesQueryVersion.restype = ctypes.c_int

    libXfixes.XFixesHideCursor.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    libXfixes.XFixesHideCursor.restype = None
    libXfixes.XFixesShowCursor.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    libXfixes.XFixesShowCursor.restype = None
    
    _X11_DISPLAY = libX11.XOpenDisplay(None)  # connection to the X server
    _X11_ROOT = libX11.XDefaultRootWindow(_X11_DISPLAY)  # the root window handle
    _major = ctypes.c_int()  # to check if the XFixes extension is present
    _minor = ctypes.c_int()
    _XFIXES_OK = libXfixes.XFixesQueryVersion(_X11_DISPLAY, ctypes.byref(_major), ctypes.byref(_minor)) != 0

    def hide_cursor_everywhere():
        if not _X11_DISPLAY or not _XFIXES_OK:
            return
        libXfixes.XFixesHideCursor(_X11_DISPLAY, _X11_ROOT)
        # flush the request buffer to ensure the command is sent immediately
        libX11.XFlush(_X11_DISPLAY)

    def restore_default_cursors():
        if not _X11_DISPLAY or not _XFIXES_OK:
            return
        libXfixes.XFixesShowCursor(_X11_DISPLAY, _X11_ROOT)
        # flush the request buffer to ensure the command is sent immediately
        libX11.XFlush(_X11_DISPLAY)

else:
    # macOS implementation using ApplicationServices
    import ctypes
    _CURSOR_HIDDEN = False
    try:
        _app_services = ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
            )

        _app_services.CGMainDisplayID.restype = ctypes.c_uint32
        _app_services.CGDisplayHideCursor.argtypes = [ctypes.c_uint32]
        _app_services.CGDisplayHideCursor.restype = ctypes.c_int
        _app_services.CGDisplayShowCursor.argtypes = [ctypes.c_uint32]
        _app_services.CGDisplayShowCursor.restype = ctypes.c_int

        def hide_cursor_everywhere():
            global _CURSOR_HIDDEN
            if _CURSOR_HIDDEN:
                return
            display_id = _app_services.CGMainDisplayID()
            _app_services.CGDisplayHideCursor(display_id)
            _CURSOR_HIDDEN = True

        def restore_default_cursors():
            global _CURSOR_HIDDEN
            if not _CURSOR_HIDDEN:
                return
            display_id = _app_services.CGMainDisplayID()
            _app_services.CGDisplayShowCursor(display_id)
            _CURSOR_HIDDEN = False
    except Exception:
        def hide_cursor_everywhere():
            pass
        def restore_default_cursors():
            pass


# 2) Mouse acceleration functions
# ---------------------------------------

if sys.platform.startswith("win"):
    import winreg

    # Constantes Windows
    SPI_SETMOUSE = 0x0004
    SPIF_SENDCHANGE = 0x02

    _ORIGINAL_MOUSE_ACCEL = None
    _HONOR_ACCEL_BACKUP = None


    def disable_mouse_acceleration():
        # Disable Windows mouse acceleration (set all thresholds to 0)
        # and remember the original settings for later restoration
        global _ORIGINAL_MOUSE_ACCEL
        global _HONOR_ACCEL_BACKUP
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
                    
                # Some touchpads ignore the usual mouse acceleration registry settings
                # To make them respect these settings we try to enable "HonorMouseAccelSetting" in the registry
                # This tells the touchpad driver to follow the normal mouse acceleration config
                # Note: This setting only works on some systems depending on the driver used
                try:
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\PrecisionTouchPad") as key:
                        _HONOR_ACCEL_BACKUP, _ = winreg.QueryValueEx(key, "HonorMouseAccelSetting")
                except FileNotFoundError:
                    _HONOR_ACCEL_BACKUP = None
                try:
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\PrecisionTouchPad", 0, winreg.KEY_SET_VALUE) as key:
                        winreg.SetValueEx(key, "HonorMouseAccelSetting", 0, winreg.REG_DWORD, 1)
                except FileNotFoundError:
                    pass

                # Apply the registry settings immediately to the running system
                params = (ctypes.c_int * 3)(0, 0, 0)
                ctypes.windll.user32.SystemParametersInfoW(SPI_SETMOUSE, 0, ctypes.byref(params), SPIF_SENDCHANGE)


    def restore_mouse_acceleration():
        # Restore the previously saved Windows mouse acceleration settings
        global _ORIGINAL_MOUSE_ACCEL
        global _HONOR_ACCEL_BACKUP
        if _ORIGINAL_MOUSE_ACCEL:
            speed, th1, th2 = _ORIGINAL_MOUSE_ACCEL

            # Write new mouse acceleration settings into the registry
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Control Panel\Mouse", 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "MouseSpeed", 0, winreg.REG_SZ, speed)
                winreg.SetValueEx(key, "MouseThreshold1", 0, winreg.REG_SZ, th1)
                winreg.SetValueEx(key, "MouseThreshold2", 0, winreg.REG_SZ, th2)
                
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\PrecisionTouchPad", 0, winreg.KEY_SET_VALUE) as key:
                    if _HONOR_ACCEL_BACKUP is not None:
                        winreg.SetValueEx(key, "HonorMouseAccelSetting", 0, winreg.REG_DWORD, int(_HONOR_ACCEL_BACKUP))
                    else:
                        winreg.DeleteValue(key, "HonorMouseAccelSetting")
            except FileNotFoundError:
                pass

            # Apply the registry settings immediately to the running system
            params = (ctypes.c_int * 3)(int(th1), int(th2), int(speed))
            ctypes.windll.user32.SystemParametersInfoW(SPI_SETMOUSE, 0, ctypes.byref(params), SPIF_SENDCHANGE)

            _ORIGINAL_MOUSE_ACCEL = None
            _HONOR_ACCEL_BACKUP = None

elif sys.platform.startswith("linux"):
    import subprocess
    import re
    _AFFECTED_DEVICES = []

    def _list_pointer_devices():
        result = subprocess.run(['xinput', '--list'], stdout=subprocess.PIPE, text=True)
        devices = {}
        for line in result.stdout.split('\n'):
            if 'pointer' in line.lower():
                match_id = re.search(r'id=(\d+)', line)
                match_name = re.search(r'^\s*(.+?)\s+id=', line)
                if match_id and match_name:
                    name = match_name.group(1).strip()
                    dev_id = match_id.group(1)
                    devices[name] = dev_id
        return devices

    def _get_props(device_id):
        result = subprocess.run(['xinput', '--list-props', device_id], stdout=subprocess.PIPE, text=True)
        return result.stdout

    def _has_libinput_profile(device_id):
        return 'libinput Accel Profile Enabled' in _get_props(device_id)

    def _has_evdev_profile(device_id):
        return 'Device Accel Profile' in _get_props(device_id)

    def _set_libinput_profile(device_id, adaptive: int, flat: int):
        subprocess.run(['xinput', '--set-prop', device_id, 'libinput Accel Profile Enabled', str(adaptive), str(flat)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _set_evdev_profile(device_id, profile_value: int):
        subprocess.run(['xinput', '--set-prop', device_id, 'Device Accel Profile', str(profile_value)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def disable_mouse_acceleration():
        global _AFFECTED_DEVICES
        _AFFECTED_DEVICES.clear()
        for name, dev_id in _list_pointer_devices().items():
            if _has_libinput_profile(dev_id):
                _set_libinput_profile(dev_id, 0, 1)
                _AFFECTED_DEVICES.append((dev_id, "libinput"))
            elif _has_evdev_profile(dev_id):
                _set_evdev_profile(dev_id, 0)
                _AFFECTED_DEVICES.append((dev_id, "evdev"))

    def restore_mouse_acceleration():
        global _AFFECTED_DEVICES
        for dev_id, method in _AFFECTED_DEVICES:
            if method == "libinput":
                _set_libinput_profile(dev_id, 1, 0)
            elif method == "evdev":
                _set_evdev_profile(dev_id, 1)
        _AFFECTED_DEVICES.clear()

else:
    # For macOS 
    import subprocess

    _MAC_MOUSE_ACCEL_BACKUP = None

    def _mac_read_mouse_scaling():
        """
        Read the current mac OS mouse scaling value from .GlobalPreferences.
        Returns:
            float | None
        """
        try:
            result = subprocess.run(
                ["defaults", "read", ".GlobalPreferences", "com.apple.mouse.scaling"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except Exception:
            return None
    def _mac_write_mouse_scaling(value):
        """
        Write macOS mouse scaling value to .GlovalPreferences.
        Returns:
            bool
        """
        try:
            subprocess.run(
                ["defaults", "write", ".GlobalPreferences", "com.apple.mouse.scaling", "-float", str(value)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            # Try to refresh preference daemon so the change is noticed sooner
            subprocess.run(
                ["killall", "cfprefsd"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except Exception:
            return False
    
    def disable_mouse_acceleration():
        """
        Best-effort macOS implementation.

        Historical command-line pratice is to set:
            com.apple.mouse.scaling = -1
        to disable acceleration, ut behavior can vary across macOS versions.
        """
        global _MAC_MOUSE_ACCEL_BACKUP

        if sys.platform != "darwin":
            return
  
        if _MAC_MOUSE_ACCEL_BACKUP is None: # read current acceleration
            _MAC_MOUSE_ACCEL_BACKUP = _mac_read_mouse_scaling()
        
        _mac_write_mouse_scaling(-1.0)

    def restore_mouse_acceleration():
        """
        Restore the previously saved macOS mouse scaling value.
        """
        global _MAC_MOUSE_ACCEL_BACKUP

        if sys.platform != "darwin":
            return

        if _MAC_MOUSE_ACCEL_BACKUP is not None:
            _mac_write_mouse_scaling(_MAC_MOUSE_ACCEL_BACKUP)
            _MAC_MOUSE_ACCEL_BACKUP = None
