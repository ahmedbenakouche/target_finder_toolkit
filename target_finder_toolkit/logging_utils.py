import atexit
import json
import time
from datetime import datetime
from pathlib import Path


def make_default_log_path(project_root: Path, technique: str) -> Path:
    logs_dir = Path(project_root) / "test_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = technique.replace(" ", "_").lower()
    return logs_dir / f"{stamp}_{safe_name}.jsonl"


class SessionLogger:
    def __init__(self, log_file: str | Path, *, cursor_hz: float = 30.0):
        self.path = Path(log_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._closed = False
        self._session_ended = False
        self._start_monotonic = time.monotonic()
        self._cursor_interval = 1.0 / max(float(cursor_hz), 1.0)
        self._last_cursor_at = 0.0
        self._atexit_registered = False
        atexit.register(self._finalize_at_exit)
        self._atexit_registered = True

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_monotonic

    def _write(self, payload: dict):
        if self._closed:
            return
        payload.setdefault("t", round(self._elapsed(), 6))
        self._fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._fh.flush()

    def log_session_start(self, **fields):
        self._write({"type": "session_start", **fields})

    def log_session_end(self, **fields):
        if self._session_ended:
            return
        self._session_ended = True
        self._write({"type": "session_end", **fields})

    def _finalize_at_exit(self):
        if self._closed:
            return
        if not self._session_ended:
            self.log_session_end(reason="process_exit")
        self.close(finalize=False)

    def log_cursor_sample(
        self,
        *,
        raw_x: float,
        raw_y: float,
        filtered_x: float,
        filtered_y: float,
        **fields,
    ):
        now = time.monotonic()
        if now - self._last_cursor_at < self._cursor_interval:
            return
        self._last_cursor_at = now
        self._write(
            {
                "type": "cursor_sample",
                "raw": [round(raw_x, 3), round(raw_y, 3)],
                "filtered": [round(filtered_x, 3), round(filtered_y, 3)],
                **fields,
            }
        )

    def log_click(self, **fields):
        self._write({"type": "click", **fields})

    def log_detection_change(self, detections, added, removed):
        compact = []
        for det in detections:
            compact.append(
                {
                    "id": det.get("id"),
                    "x": det.get("x"),
                    "y": det.get("y"),
                    "w": det.get("width"),
                    "h": det.get("height"),
                    "score": round(float(det.get("score", 0.0)), 4),
                    "class": det.get("class_name"),
                }
            )
        self._write(
            {
                "type": "detection_change",
                "detections": compact,
                "added_ids": [det.get("id") for det in added],
                "removed_ids": [det.get("id") for det in removed],
            }
        )

    def close(self, *, finalize: bool = True):
        if self._closed:
            return
        if finalize and not self._session_ended:
            self.log_session_end(reason="logger_close")
        try:
            self._fh.close()
        except Exception:
            pass
        self._closed = True
        if self._atexit_registered:
            try:
                atexit.unregister(self._finalize_at_exit)
            except Exception:
                pass
            self._atexit_registered = False
