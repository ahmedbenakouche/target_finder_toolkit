"""
Bubble Cursor Demo
==================

This module demonstrates the **Bubble Cursor** interaction technique
using the TargetFinder toolkit.

Notes
-----
- This script is for demonstration purposes only.
- It is **not part of the core TargetFinder API**, but shows how the
  toolkit can be used for novel interaction techniques.
"""


import os
import sys
import time
import signal
import atexit
import threading
import numpy as np
import cv2
import mss
from ultralytics import YOLO
from PyQt6 import QtWidgets, QtGui, QtCore
import math
import pyautogui
from pynput import keyboard, mouse
import argparse
from target_finder_toolkit.targetfinder import CLASS_NAMES, TargetFinder
from target_finder_toolkit.mouse_utils import hide_cursor_everywhere, restore_default_cursors
from target_finder_toolkit.filters import FILTER_OPTIONS, PointFilter2D
from target_finder_toolkit.logging_utils import SessionLogger

__all__ = ["bubble_cursor", "main"]


_CURSOR_RESTORE_REGISTERED = False

class BubbleCursor(QtWidgets.QWidget):
    """
    PyQt overlay widget implementing the Bubble Cursor technique.

    Principle
    ---------
    At each frame, the system computes the intersecting distance (IntD)
    from the pointer to all detected widgets. The bubble radius is set
    to encompass the closest widget but not overlap the second closest.
    A visual "bubble" is drawn around the pointer and the target.

    Parameters
    ----------
    detector : TargetFinder
        Detector providing widget bounding boxes (logical coordinates).

    Notes
    -----
    - The overlay is transparent and always on top.
    - Clicking outside a widget inside the bubble is redirected
      to the nearest detected target.
    - The real cursor is hidden and replaced by a drawn "fake" cursor.
    """
    def __init__(self, detector: TargetFinder, cursor_filter=None, logger=None):
        super().__init__()
        self.detector = detector
        detector.overlay_window = self
        self.cursor_filter = cursor_filter
        self.logger = logger
        self._last_target = None  # to store the active target
        self._last_target_info = None
        self._pending_click_target = None
        self._pending_click_target_info = None
        self._pending_click_enabled = False
        self._mouse_listener = None
        self._last_rehide_at = 0.0
        self.bubble_enabled = True
        self._start_mouse_listener()
        self._start_keyboard_listener()

        # Full-screen geometry (Qt DPI-aware)
        geom = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(geom)

        # Window flags for frameless, always-on-top, click-through & transparent
        flags = (
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.WindowTransparentForInput
        )
        if sys.platform != "darwin":
            flags |= QtCore.Qt.WindowType.Tool
        if sys.platform.startswith("linux"):
            flags |= QtCore.Qt.WindowType.X11BypassWindowManagerHint

        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        # Start detection thread
        self.detector.start()

        # refresh to update the overlay
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(10) # 10 ms
        self._cursor_refresh_timer = None
        if sys.platform == "darwin":
            self._cursor_refresh_timer = QtCore.QTimer(self)
            self._cursor_refresh_timer.timeout.connect(self._rehide_cursor)
            self._cursor_refresh_timer.start(30)

    def _resolve_target_info(self, tx, ty, w, h, cls_id, score):
        target_info = self.detector.find_detection_by_geometry(
            tx, ty, w, h, class_id=cls_id
        )
        if target_info is not None:
            return target_info

        target_info = self.detector.find_detection_for_point(
            float(tx),
            float(ty),
            include_text=(cls_id == 3),
            fallback_nearest=True,
        )
        if target_info is not None:
            return target_info

        return {
            "id": None,
            "x": int(round(tx - w / 2)),
            "y": int(round(ty - h / 2)),
            "w": int(round(w)),
            "h": int(round(h)),
            "score": round(float(score), 4),
            "class": CLASS_NAMES.get(int(cls_id), str(cls_id)),
            "class_id": int(cls_id),
        }

    def _screen_for_point(self, x: int, y: int):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return QtWidgets.QApplication.primaryScreen()
        return app.screenAt(QtCore.QPoint(int(x), int(y))) or QtWidgets.QApplication.primaryScreen()

    def _snapshot_click_target(self):
        self._pending_click_target = tuple(self._last_target) if self._last_target else None
        self._pending_click_target_info = (
            dict(self._last_target_info) if self._last_target_info is not None else None
        )
        self._pending_click_enabled = bool(self.bubble_enabled)

    def _take_click_target_snapshot(self):
        target = self._pending_click_target
        target_info = self._pending_click_target_info
        enabled = self._pending_click_enabled
        self._pending_click_target = None
        self._pending_click_target_info = None
        self._pending_click_enabled = False
        return target, target_info, enabled

    def _resolve_click_target_for_log(self, x, y, fallback_info=None, *, fallback_nearest=False):
        if fallback_info is not None:
            return dict(fallback_info)
        target_info = self.detector.find_detection_for_point(
            float(x),
            float(y),
            include_text=True,
            fallback_nearest=fallback_nearest,
        )
        return target_info

    def _in_system_reserved_area(self, x: int, y: int) -> bool:
        if sys.platform != "darwin":
            return False
        screen = self._screen_for_point(x, y)
        if screen is None:
            return False
        point = QtCore.QPoint(int(x), int(y))
        return not screen.availableGeometry().contains(point)

    # === Paint ===
    def paintEvent(self, event):
        detections = self.detector.get_detections()
        if not self.bubble_enabled or not detections:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Pointer position (logical coordinates)
        pos = QtGui.QCursor.pos()
        if self.cursor_filter is None:
            cx, cy = pos.x(), pos.y()
            raw_x, raw_y = float(cx), float(cy)
        else:
            raw_x, raw_y = float(pos.x()), float(pos.y())
            cx, cy = raw_x, raw_y
            cx, cy = self.cursor_filter.filter(raw_x, raw_y)

        text_margin = 20
        cursor_on_non_text = False
        for x, y, w, h, score, cls_id in detections:
            if cls_id != 3 and x <= cx <= x + w and y <= cy <= y + h:
                cursor_on_non_text = True
                break
        if not cursor_on_non_text:
            for x, y, w, h, score, cls_id in detections:
                if cls_id == 3:
                    if ((x - text_margin) <= cx <= (x + w + text_margin) and
                            (y - text_margin) <= cy <= (y + h + text_margin)):
                        self._last_target = None
                        self._last_target_info = None
                        self._draw_fake_cursor(painter, cx, cy)
                        painter.end()
                        return

        # Compute distances from pointer to each box edge
        distances = []
        for x, y, w, h, score, cls_id in detections:
            cx_box = x + w/2
            cy_box = y + h/2

            # Intersecting Distance (IntD)
            dx = max(0.0, abs(cx - cx_box) - w/2)
            dy = max(0.0, abs(cy - cy_box) - h/2)
            IntD = math.hypot(dx, dy)

            distances.append((IntD, cx_box, cy_box, w, h, score, cls_id))

        # Draw bubble only when not hovering over text
        if not distances:
            self._last_target = None
            self._last_target_info = None
            self._draw_fake_cursor(painter,cx,cy)
            painter.end()
            return

        distances.sort(key=lambda t: t[0])
        IntD1, tx, ty, w, h, nearest_score, nearest_cls_id = distances[0]

        if nearest_cls_id == 3:
            self._last_target = None
            self._last_target_info = None
            self._draw_fake_cursor(painter,cx,cy)
            painter.end()
            return
        self._last_target = (
        tx / self.detector.sx, ty / self.detector.sy, w / self.detector.sx, h / self.detector.sy)
        self._last_target_info = self._resolve_target_info(
            tx, ty, w, h, nearest_cls_id, nearest_score
        )

        # Containment Distance (ConD1)
        x = tx - w / 2
        y = ty - h / 2
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        ConD1 = max([math.hypot(cx - px, cy - py) for (px, py) in corners])

        if len(distances) > 1:
            IntD2 = distances[1][0]
            radius = min(ConD1, IntD2)
            if radius == IntD2:
                gap = 0.01 * radius
                radius = int(radius - gap)  # to avoid touching the second target
        else:
            radius = ConD1

        # build main and env bubble with different centers
        main_path = QtGui.QPainterPath()
        main_path.addEllipse(cx - radius, cy - radius, 2 * radius, 2 * radius)

        # env bubble centered on widget center
        env_path = QtGui.QPainterPath()
        t = 1  # t interpolates between 0 = sharp‑cornered rectangle / 1 = fully rounded (ellipse‑like)
        r = min(w, h) / 2 * t  # corner radius
        d = math.hypot(r, r) - r
        env_path.addRoundedRect(tx - w/2 -d, ty - h/2 - d, w + 2*d, h + 2*d, r + d, r + d)

        # Or we can also use a circle around it or a ellipse
        # reinforce radius: half max dimension of widget = max(w, h) / 2.0
        # ew, eh = w * math.sqrt(2), h * math.sqrt(2)
        # env_path.addEllipse(tx - ew/2, ty - eh/2, ew, eh)

        union_path = main_path.united(env_path)
        pen = QtGui.QPen(QtGui.QColor(0, 255, 0, 200), 3)
        painter.setPen(pen)
        painter.drawPath(union_path)
        self._draw_fake_cursor(painter,cx,cy)
        if self.logger is not None:
            self.logger.log_cursor_sample(
                raw_x=raw_x,
                raw_y=raw_y,
                filtered_x=cx,
                filtered_y=cy,
                technique="bubble",
                filter_name=self.cursor_filter.filter_name if self.cursor_filter is not None else "none",
                bubble_radius=round(float(radius), 3),
                has_target=self._last_target is not None,
            )
        painter.end()
        return


    # FAKE CURSOR
    def _draw_fake_cursor(self,painter,cx,cy):
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 2)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        radius_cursor = 6
        center = QtCore.QPointF(cx, cy)
        painter.drawEllipse(center, radius_cursor, radius_cursor)
        line_len = 4
        gap = radius_cursor + 1
        painter.drawLine(QtCore.QLineF(cx, cy - gap - line_len, cx, cy - gap))
        painter.drawLine(QtCore.QLineF(cx, cy + gap, cx, cy + gap + line_len))
        painter.drawLine(QtCore.QLineF(cx + gap, cy, cx + gap + line_len, cy))
        painter.drawLine(QtCore.QLineF(cx - gap - line_len, cy, cx - gap, cy))
    

    # === Toggle bubble on/off ===
    @QtCore.pyqtSlot()
    def toggle_bubble(self):
        self.bubble_enabled = not self.bubble_enabled
        if self.bubble_enabled:
            self.detector.start()
            hide_cursor_everywhere()
        else:
            self.detector.stop()
            restore_default_cursors()
        self.update()  # repaint immediately

    # === Quit ===
    @QtCore.pyqtSlot()
    def stop_and_quit(self):
        self.bubble_enabled = False
        self.detector.stop()
        if self._mouse_listener is not None:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
            self._mouse_listener = None
        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            self._keyboard_listener = None
        if self._cursor_refresh_timer is not None:
            self._cursor_refresh_timer.stop()
        restore_default_cursors()
        if self.logger is not None:
            self.logger.log_session_end(reason="quit")
            self.logger.close()
        self.close()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()

    def closeEvent(self, event):
        self.bubble_enabled = False
        self.detector.stop()
        if self._mouse_listener is not None:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
            self._mouse_listener = None
        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            self._keyboard_listener = None
        if self._cursor_refresh_timer is not None:
            self._cursor_refresh_timer.stop()
        restore_default_cursors()
        super().closeEvent(event)

    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == 'b':
                    QtCore.QMetaObject.invokeMethod(self, "toggle_bubble", QtCore.Qt.ConnectionType.QueuedConnection)
                elif key.char == 'q':
                    QtCore.QMetaObject.invokeMethod(self, "stop_and_quit", QtCore.Qt.ConnectionType.QueuedConnection)
            except AttributeError:
                pass
        self._keyboard_listener = keyboard.Listener(on_press=on_press)
        self._keyboard_listener.start()

    @QtCore.pyqtSlot()
    def _rehide_cursor(self):
        if not self.bubble_enabled:
            return
        hide_cursor_everywhere()
        if sys.platform == "darwin":
            QtCore.QTimer.singleShot(0, hide_cursor_everywhere)
            QtCore.QTimer.singleShot(25, hide_cursor_everywhere)
            QtCore.QTimer.singleShot(75, hide_cursor_everywhere)

    # === Global mouse listener + click simulation ===
    def _start_mouse_listener(self):
        if sys.platform != "darwin":
            def on_click(x, y, button, pressed):
                if pressed and button == button.left:
                    QtCore.QMetaObject.invokeMethod(
                        self,
                        "_simulate_click",
                        QtCore.Qt.ConnectionType.QueuedConnection,
                        QtCore.Q_ARG(int, x),
                        QtCore.Q_ARG(int, y),
                    )
            self._mouse_listener = mouse.Listener(on_click=on_click)
            self._mouse_listener.start()
            return

        def queue_rehide(force=False):
            now = time.monotonic()
            if not force and now - self._last_rehide_at < 0.008:
                return
            self._last_rehide_at = now
            QtCore.QMetaObject.invokeMethod(
                self,
                "_rehide_cursor",
                QtCore.Qt.ConnectionType.QueuedConnection,
            )

        def on_move(x, y):
            queue_rehide()

        def on_click(x, y, button, pressed):
            if button == button.left:
                queue_rehide(force=True)
            if pressed and button == button.left:
                self._snapshot_click_target()
                if self._in_system_reserved_area(x, y):
                    return
                # simulate in the Qt thread
                QtCore.QMetaObject.invokeMethod(self, "_simulate_click", QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(int, x), QtCore.Q_ARG(int, y))
        self._mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
        self._mouse_listener.start()

    @QtCore.pyqtSlot(int, int)
    def _simulate_click(self, orig_x, orig_y):
        click_target, click_target_info, click_enabled = self._take_click_target_snapshot()
        if click_target is None:
            click_target = self._last_target
        if click_target_info is None:
            click_target_info = self._last_target_info
        if not click_enabled:
            click_enabled = self.bubble_enabled

        if self._in_system_reserved_area(orig_x, orig_y):
            if self.logger is not None:
                self.logger.log_click(
                    technique="bubble",
                    raw=[orig_x, orig_y],
                    effective=[orig_x, orig_y],
                    redirected=False,
                    target=None,
                )
            return
        if not (click_target and click_enabled):
            if self.logger is not None:
                self.logger.log_click(
                    technique="bubble",
                    raw=[orig_x, orig_y],
                    effective=[orig_x, orig_y],
                    redirected=False,
                    target=None,
                )
            return

        tx, ty, w, h = click_target

        # If click inside detected rectangle let it pass
        if tx-w/2 <= orig_x <= tx+w/2 and ty-h/2 <= orig_y <= ty+h/2:
            if self.logger is not None:
                self.logger.log_click(
                    technique="bubble",
                    raw=[orig_x, orig_y],
                    effective=[orig_x, orig_y],
                    redirected=False,
                    target=self._resolve_click_target_for_log(
                        orig_x,
                        orig_y,
                        click_target_info,
                        fallback_nearest=False,
                    ),
                )
            return

        # Otherwise simulate target click
        self._mouse_listener.stop()  # stop listener
        try:
            pyautogui.mouseUp(button='left')  # simulate button release
            pyautogui.moveTo(tx, ty) # move to and click the targeted widget
            pyautogui.click()
            pyautogui.moveTo(orig_x, orig_y) # move the mouse back to its original position
        except pyautogui.FailSafeException:
            # to prevent the script from crashing when the mouse hits a corner
            pass
        finally:
            if self.logger is not None:
                self.logger.log_click(
                    technique="bubble",
                    raw=[orig_x, orig_y],
                    effective=[round(tx, 3), round(ty, 3)],
                    redirected=True,
                    target=self._resolve_click_target_for_log(
                        tx,
                        ty,
                        click_target_info,
                        fallback_nearest=True,
                    ),
                )
            self._rehide_cursor()
            self._start_mouse_listener() # restart the listener


def bubble_cursor(detector: TargetFinder, cursor_filter=None, logger=None):
    """Launch the Bubble Cursor overlay.

    This replaces the system cursor with a dynamic "bubble" that
    expands to always contain the closest widget detected by
    :class:`TargetFinder`.

    Args:
        detector (TargetFinder): Initialized detector (YOLO model loaded).

    Returns:
        None: Blocks until the Qt application is closed.
    """
    global _CURSOR_RESTORE_REGISTERED
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    if not _CURSOR_RESTORE_REGISTERED:
        atexit.register(restore_default_cursors)
        _CURSOR_RESTORE_REGISTERED = True
    ov = BubbleCursor(detector, cursor_filter=cursor_filter, logger=logger)
    ov.show()
    if sys.platform == "darwin":
        QtCore.QTimer.singleShot(0, hide_cursor_everywhere)
        QtCore.QTimer.singleShot(25, hide_cursor_everywhere)
    else:
        hide_cursor_everywhere()
    signal.signal(signal.SIGINT, lambda sig, frame: QtWidgets.QApplication.quit())
    exit_code = app.exec()
    restore_default_cursors()
    if logger is not None:
        logger.log_session_end(reason="app_exit")
        logger.close()
    sys.exit(exit_code)


# CLI usage
def main():
    """Command-line entry point for the Bubble Cursor demo.

    CLI arguments:
        - -model-path (str, optional): Path to YOLO .pt weights.
            Defaults to ``yolo26s_1280.pt`` in the package.
        - -change-thresh (int, optional): Threshold for screen change detection.
            Higher = less sensitive. Default = ``100``.
        - -capture-interval (float, optional): Delay in seconds between captures.
            Lower = higher refresh rate, more CPU/GPU usage. Default = ``1/30``.
        - -confidence (float, optional): YOLO confidence threshold in ``[0, 1]``.
            Default = ``0.28``.
        - -iou (float, optional): IoU threshold for YOLO NMS in ``[0, 1]``.
            Controls overlap merging. Default = ``0.3``.

    Keyboard shortcuts:
        - **b**: Toggle between the Bubble Cursor and the normal system cursor.
        - **q**: Quit the program.

    **Example:** ``bubblecursor --change-thresh 200 --confidence 0.3 --iou 0.4``

    Returns:
        Starts the Qt event loop until exit.
    """
    parser = argparse.ArgumentParser(description="Launch the BubbleCursor overlay")
    parser.add_argument('--model-path', default=None, help="Path to the YOLO model .pt file")
    parser.add_argument('--change-thresh', type=int, default=100, help="Threshold for detecting screen changes")
    parser.add_argument('--capture-interval', type=float, default=1 / 30, help="Interval between screen captures (in seconds)")
    parser.add_argument('--confidence', type=float, default=0.28, help="YOLO confidence threshold (0.0–1.0)")
    parser.add_argument('--iou', type=float, default=0.3, help="YOLO IoU threshold for NMS (0.0–1.0)")
    parser.add_argument('--filter', choices=sorted(FILTER_OPTIONS.keys()), default="none", help="Optional pointer filter")
    parser.add_argument('--log-file', default=None, help="Optional JSONL log file path")
    parser.add_argument('--log-cursor-hz', type=float, default=30.0, help="Cursor sampling rate for logging")
    args = parser.parse_args()

    if args.model_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(here, "yolo26s_1280.pt")

    det = TargetFinder(args.model_path, args.change_thresh, args.capture_interval, args.confidence, args.iou)
    cursor_filter = PointFilter2D(args.filter) if args.filter != "none" else None
    logger = SessionLogger(args.log_file, cursor_hz=args.log_cursor_hz) if args.log_file else None
    if logger is not None:
        logger.log_session_start(
            technique="bubble",
            filter_name=args.filter,
            model_path=args.model_path,
            change_thresh=args.change_thresh,
            capture_interval=args.capture_interval,
            confidence=args.confidence,
            iou=args.iou,
        )
        det.set_callback(lambda dets, added, removed, _frame: logger.log_detection_change(dets, added, removed))
    bubble_cursor(det, cursor_filter=cursor_filter, logger=logger)

if __name__ == "__main__":
    main()
