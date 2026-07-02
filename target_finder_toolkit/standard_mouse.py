"""Standard mouse tester with an optional pointer filter.

This tester-only mode runs the native system cursor without any target-aware
assistance. If a filter is selected, it applies that filter on top of the
standard cursor for pilot testing.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

from PyQt6 import QtCore, QtGui, QtWidgets
from pynput import keyboard, mouse

from target_finder_toolkit.filters import add_filter_arguments, filter_kwargs_from_args, PointFilter2D
from target_finder_toolkit.logging_utils import SessionLogger
from target_finder_toolkit.targetfinder import TargetFinder


class StandardMouseRunner(QtCore.QObject):
    def __init__(self, args, logger: SessionLogger | None = None):
        super().__init__()
        self.args = args
        self.logger = logger
        self.detector = TargetFinder(
            args.model_path,
            args.change_thresh,
            args.capture_interval,
            args.confidence,
            args.iou,
        )
        self.cursor_filter = PointFilter2D(args.filter, **filter_kwargs_from_args(args))
        self._mouse_listener = None
        self._keyboard_listener = None
        self._ignore_warp_until = 0.0
        self._last_filtered = None
        self._stop_reason = "app_exit"

    def start(self):
        pos = QtGui.QCursor.pos()
        initial = (float(pos.x()), float(pos.y()))
        self.cursor_filter.reset(*initial)
        self._last_filtered = initial
        if self.logger is not None:
            self.logger.log_session_start(
                technique="standard_mouse",
                filter_name=self.cursor_filter.filter_name,
                **self.cursor_filter.params,
                model_path=self.args.model_path,
                change_thresh=self.args.change_thresh,
                capture_interval=self.args.capture_interval,
                confidence=self.args.confidence,
                iou=self.args.iou,
                detection_source="yolo",
            )
            self.detector.set_callback(
                lambda dets, added, removed, _frame: self.logger.log_detection_change(dets, added, removed)
            )
        self.detector.start()
        self._start_mouse_listener()
        self._start_keyboard_listener()

    def _start_mouse_listener(self):
        def on_move(x, y):
            if time.monotonic() < self._ignore_warp_until:
                return
            QtCore.QMetaObject.invokeMethod(
                self,
                "_apply_filtered_position",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(float, float(x)),
                QtCore.Q_ARG(float, float(y)),
            )

        def on_click(x, y, button, pressed):
            if pressed and self.logger is not None:
                target = self.detector.find_detection_for_point(
                    float(x),
                    float(y),
                    include_text=True,
                    fallback_nearest=False,
                )
                self.logger.log_click(
                    technique="standard_mouse",
                    button=str(button),
                    raw=[round(float(x), 3), round(float(y), 3)],
                    effective=[round(float(x), 3), round(float(y), 3)],
                    redirected=False,
                    filtered=(
                        [round(self._last_filtered[0], 3), round(self._last_filtered[1], 3)]
                        if self._last_filtered is not None
                        else None
                    ),
                    target=target,
                )

        self._mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
        self._mouse_listener.start()

    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == "q":
                    self._stop_reason = "q_pressed"
                    QtCore.QMetaObject.invokeMethod(
                        QtWidgets.QApplication.instance(),
                        "quit",
                        QtCore.Qt.ConnectionType.QueuedConnection,
                    )
            except AttributeError:
                pass

        self._keyboard_listener = keyboard.Listener(on_press=on_press)
        self._keyboard_listener.start()

    @QtCore.pyqtSlot(float, float)
    def _apply_filtered_position(self, raw_x: float, raw_y: float):
        filtered_x, filtered_y = self.cursor_filter.filter(raw_x, raw_y)
        self._last_filtered = (filtered_x, filtered_y)
        if self.logger is not None:
            self.logger.log_cursor_sample(
                raw_x=raw_x,
                raw_y=raw_y,
                filtered_x=filtered_x,
                filtered_y=filtered_y,
                technique="standard_mouse",
                filter_name=self.cursor_filter.filter_name,
                **self.cursor_filter.params,
                detection_count=len(self.detector.get_detections()),
            )
        if self.cursor_filter.enabled:
            self._ignore_warp_until = time.monotonic() + 0.006
            QtGui.QCursor.setPos(int(round(filtered_x)), int(round(filtered_y)))

    def stop(self):
        try:
            self.detector.stop()
        except Exception:
            pass
        for listener in (self._mouse_listener, self._keyboard_listener):
            if listener is None:
                continue
            try:
                listener.stop()
            except Exception:
                pass
        if self.logger is not None:
            self.logger.log_session_end(reason=self._stop_reason)
            self.logger.close()


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run a standard mouse tester with an optional pointer filter.")
    parser.add_argument("--model-path", default=None, help="Path to the YOLO model .pt file")
    parser.add_argument("--change-thresh", type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument("--capture-interval", type=float, default=1 / 30, help="Interval between screen captures in seconds")
    parser.add_argument("--confidence", type=float, default=0.28, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.3, help="YOLO IoU threshold for NMS")
    add_filter_arguments(parser)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-cursor-hz", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.model_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(here, "yolo26s_1280.pt")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    logger = SessionLogger(args.log_file, cursor_hz=args.log_cursor_hz) if args.log_file else None
    runner = StandardMouseRunner(args, logger=logger)

    def handle_signal(_sig, _frame):
        runner._stop_reason = "signal_interrupt"
        app.quit()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    runner.start()
    try:
        return app.exec()
    finally:
        runner.stop()


if __name__ == "__main__":
    raise SystemExit(main())
