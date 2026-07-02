"""Windows-only subprocess cleanup helpers."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes


JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def attach_windows_kill_on_close_job(proc: subprocess.Popen) -> subprocess.Popen:
    """Attach a Windows subprocess to a job killed when the parent exits.

    Qt overlay processes can otherwise survive after the main experiment process
    crashes on Windows. This is intentionally a no-op outside Windows.
    """

    if not sys.platform.startswith("win"):
        return proc
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.INT,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        job_handle = kernel32.CreateJobObjectW(None, None)
        if not job_handle:
            return proc

        info = _JobObjectExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job_handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            kernel32.CloseHandle(job_handle)
            return proc

        process_handle = getattr(proc, "_handle", None)
        if process_handle is None:
            kernel32.CloseHandle(job_handle)
            return proc
        ok = kernel32.AssignProcessToJobObject(job_handle, process_handle)
        if not ok:
            kernel32.CloseHandle(job_handle)
            return proc

        setattr(proc, "_target_finder_windows_job", job_handle)
    except Exception:
        return proc
    return proc


def close_windows_process_job(proc: subprocess.Popen | None) -> None:
    """Close the stored Windows job handle after the subprocess has stopped."""

    if proc is None or not sys.platform.startswith("win"):
        return
    job_handle = getattr(proc, "_target_finder_windows_job", None)
    if not job_handle:
        return
    try:
        ctypes.windll.kernel32.CloseHandle(job_handle)
    except Exception:
        pass
    try:
        setattr(proc, "_target_finder_windows_job", None)
    except Exception:
        pass
