"""Inspect the latest Ninja Cursors gaze calibration result."""

from __future__ import annotations

import argparse
import json
import math
import pathlib


DEFAULT_PATH = pathlib.Path.home() / ".target_finder_toolkit" / "eye_calibration" / "last_calibration.json"


def _as_float(value, default=0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _quality_label(mean_error_px: float, half_cursor_gap_px: float) -> str:
    if mean_error_px <= half_cursor_gap_px * 0.45:
        return "good"
    if mean_error_px <= half_cursor_gap_px * 0.75:
        return "usable"
    if mean_error_px <= half_cursor_gap_px:
        return "risky"
    return "bad"


def _format_matrix(matrix) -> str:
    if not matrix:
        return "  <missing>"
    rows = []
    for row in matrix:
        rows.append("  " + "  ".join(f"{_as_float(value): .4f}" for value in row))
    return "\n".join(rows)


def inspect_calibration(path: pathlib.Path) -> int:
    if not path.exists():
        print(f"No calibration file found: {path}")
        return 2

    data = json.loads(path.read_text())
    screen_w = _as_float(data.get("screen_w"), 1.0)
    screen_h = _as_float(data.get("screen_h"), 1.0)
    mean_error_px = _as_float(data.get("mean_error_px"))
    half_cursor_gap_px = screen_w * 0.125
    quality = _quality_label(mean_error_px, half_cursor_gap_px)
    correction = data.get("correction_values") or {}
    affine = correction.get("affine_matrix") or []
    diagnostics = data.get("diagnostics") or {}
    accepted = data.get("accepted")
    failure_reason = data.get("failure_reason")
    max_accepted_affine_error_px = data.get("max_accepted_affine_error_px")
    manual_mean_error_px = data.get("manual_mean_error_px")

    print(f"file: {path}")
    print(f"screen: {screen_w:.0f} x {screen_h:.0f}")
    print(f"points: {data.get('num_points')}")
    print(f"calibration mean error: {mean_error_px:.1f}px")
    if manual_mean_error_px is not None:
        print(f"manual gain/offset mean error: {_as_float(manual_mean_error_px):.1f}px")
    print(f"half Ninja column gap estimate: {half_cursor_gap_px:.1f}px")
    print(f"quality: {quality}")
    if accepted is not None:
        print(f"accepted: {bool(accepted)}")
    if max_accepted_affine_error_px is not None:
        print(f"max accepted affine error: {_as_float(max_accepted_affine_error_px):.1f}px")
    if failure_reason:
        print(f"failure reason: {failure_reason}")
    print(
        "manual correction: "
        f"gain=({_as_float(correction.get('gaze_gain_x')):.3f}, "
        f"{_as_float(correction.get('gaze_gain_y')):.3f}) "
        f"offset_px=({_as_float(correction.get('gaze_offset_x')):.1f}, "
        f"{_as_float(correction.get('gaze_offset_y')):.1f})"
    )
    print("affine matrix:")
    print(_format_matrix(affine))

    if diagnostics:
        print(
            "point diagnostics: "
            f"manual mean={_as_float(diagnostics.get('manual_point_mean_error_px')):.1f}px, "
            f"manual max={_as_float(diagnostics.get('manual_point_max_error_px')):.1f}px, "
            f"affine mean={_as_float(diagnostics.get('affine_point_mean_error_px')):.1f}px, "
            f"affine max={_as_float(diagnostics.get('affine_point_max_error_px')):.1f}px"
        )
        for point in diagnostics.get("points") or []:
            target_px = point.get("target_px") or [0.0, 0.0]
            manual_pred_px = point.get("manual_pred_px") or [0.0, 0.0]
            manual_error_xy_px = point.get("manual_error_xy_px") or [0.0, 0.0]
            print(
                "  "
                f"#{point.get('index')}: "
                f"target=({_as_float(target_px[0]):.0f},{_as_float(target_px[1]):.0f}) "
                f"manual=({_as_float(manual_pred_px[0]):.0f},{_as_float(manual_pred_px[1]):.0f}) "
                f"err=({_as_float(manual_error_xy_px[0]):+.0f},"
                f"{_as_float(manual_error_xy_px[1]):+.0f}) "
                f"|err|={_as_float(point.get('manual_error_px')):.1f}px "
                f"samples={point.get('sample_count_kept')}/{point.get('sample_count_raw')}"
            )
    else:
        print("point diagnostics: missing; run a fresh calibration with the updated code.")

    if accepted is False or quality in {"bad", "risky"}:
        print(
            "conclusion: calibration mapping is not accurate enough for reliable 8-cursor selection."
        )
        return 1
    print("conclusion: calibration mapping is plausible; remaining problems are likely runtime selection/noise.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=pathlib.Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    return inspect_calibration(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
