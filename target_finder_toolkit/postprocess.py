"""
postprocess.py
==============

Utility functions to consume the detection callback output of TargetFinder.

This module provides helpers for:
- Selection, sorting, and filtering of detections.
- Drawing and exporting annotated frames (bounding boxes, labels).
- Extracting and saving crops of detected widgets.

Expected data structures
------------------------
- detections: list of dicts with keys
  {id, x, y, width, height, score, class_id, class_name}
- frame: RGB image as a NumPy array.
"""

import json
from pathlib import Path
import numpy as np
import cv2


# ============== Selection / sorting / filtering ==============

def get_ids(detections):
    """
    Return sorted detection IDs (ascending).

    Parameters
    ----------
    detections : list[dict]
        Each dict is expected to have the key "id" (int).

    Returns
    -------
    list[int]
        Sorted list of IDs for detections that have a non-None "id".
    """
    return sorted(int(d["id"]) for d in detections if d.get("id") is not None)


def sort_detections(detections, key="id", reverse=False):
    """
    Return a new list of detections sorted by a given key.

    Parameters
    ----------
    detections : list[dict]
        List of detection dictionaries.
    key : str, optional
        Key to sort by. Typical values: "id", "score", "x", "y", "width", "height".
    reverse : bool, optional
        If True, sort in descending order.

    Returns
    -------
    list[dict]
        Sorted copy of the input detections.
    """
    return sorted(detections, key=lambda d: d.get(key, 0), reverse=reverse)


def summarize_change(detections, added, removed):
    """
    Build a compact, log-friendly summary of changes between two detection sets.

    Parameters
    ----------
    detections : list[dict]
        Current detections.
    added : list[dict]
        Detections that appeared (vs previous step).
    removed : list[dict]
        Detections that disappeared (vs previous step).

    Returns
    -------
    dict
        {
          "total": int,
          "added_ids": list[int],
          "removed_ids": list[int],
          "current_ids": list[int],
        }
    """
    return {
        "total": len(detections),
        "added_ids": get_ids(added),
        "removed_ids": get_ids(removed),
        "current_ids": get_ids(detections),
    }


def pretty_print_change(detections, added, removed):
    """
    Print a human-readable change summary (useful in quick logs / tests).

    Parameters
    ----------
    detections : list[dict]
    added : list[dict]
    removed : list[dict]
    """
    s = summarize_change(detections, added, removed)
    print(f"{s['total']} targets (+{len(s['added_ids'])} / -{len(s['removed_ids'])})")


# ============== Drawing / export ==============

def draw_bboxes(frame_rgb, detections, draw_ids=True, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes and labels on a copy of an RGB frame.

    The label format is: "<class_name>:<score>[#<id>]" when draw_ids=True and an id is present.
    Example: "button:0.93#12"

    Parameters
    ----------
    frame_rgb : np.ndarray
        HxWx3 RGB image.
    detections : list[dict]
        Each dict must contain x, y, width, height; may contain class_name, score, id.
    draw_ids : bool, optional
        If True, append "#<id>" to the label when present.
    color : tuple[int, int, int], optional
        Box color in BGR (OpenCV convention).
    thickness : int, optional
        Rectangle thickness.

    Returns
    -------
    np.ndarray
        Annotated RGB image (same shape as input).
    """
    out = frame_rgb.copy()
    for d in detections:
        x, y, w, h = d["x"], d["y"], d["width"], d["height"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        label = f"{d.get('class_name','?')}:{d.get('score',0):.2f}"
        if draw_ids and d.get("id") is not None:
            label += f"#{d['id']}"
        yy = max(0, y - 5)
        cv2.putText(out, label, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(out, label, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_annotated(path, frame_rgb, detections, draw_ids=True):
    """
    Save an annotated PNG image and return its path.

    Parameters
    ----------
    path : str | Path
        Output path for the PNG image.
    frame_rgb : np.ndarray
        HxWx3 RGB image.
    detections : list[dict]
        Detections to draw.
    draw_ids : bool, optional
        If True, include IDs in labels when available.

    Returns
    -------
    Path
        The output file path.
    """
    out = draw_bboxes(frame_rgb, detections, draw_ids=draw_ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)
    return path


def save_detections_json(path, detections):
    """
    Save detections (list of dicts) to a human-readable JSON file.

    Parameters
    ----------
    path : str | Path
        Output JSON path.
    detections : list[dict]
        Detection dictionaries.

    Returns
    -------
    Path
        The output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)
    return path


# ============== Crops ==============

def extract_crops(frame_rgb, detections, clamp=True):
    """
    Extract per-detection crops (RGB) from a frame.

    Parameters
    ----------
    frame_rgb : np.ndarray
        HxWx3 RGB image.
    detections : list[dict]
        Each dict must contain x, y, width, height; may contain "id".
    clamp : bool, optional
        If True, crop boxes are clamped to image bounds.

    Returns
    -------
    list[tuple[int, np.ndarray]]
        List of (det_id, crop) where det_id is the detection id or -1 if missing.
        Each crop is an HxWx3 RGB array.
    """
    H, W = frame_rgb.shape[:2]
    out = []
    for d in detections:
        x, y, w, h = int(d["x"]), int(d["y"]), int(d["width"]), int(d["height"])
        if clamp:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
        else:
            x1, y1, x2, y2 = x, y, x + w, y + h
        if x2 > x1 and y2 > y1:
            crop = frame_rgb[y1:y2, x1:x2].copy()
            out.append((int(d.get("id", -1)), crop))
    return out


def save_crops(outdir, crops, prefix="crop", ext=".png"):
    """
    Save a list of crops to a directory and return their file paths.

    The filename pattern is:
        <prefix>_<index>[_id<ID>]<ext>
    Examples:
        crop_0000.png
        crop_0001_id12.png

    Parameters
    ----------
    outdir : str | Path
        Output directory.
    crops : list[tuple[int, np.ndarray]]
        List of (det_id, crop) as returned by `extract_crops`.
    prefix : str, optional
        Filename prefix.
    ext : str, optional
        File extension (including the dot), e.g., ".png" or ".jpg".

    Returns
    -------
    list[Path]
        List of saved file paths in the same order as `crops`.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, (det_id, img) in enumerate(crops):
        stem = f"{prefix}_{idx:04d}"
        if det_id is not None and det_id >= 0:
            stem += f"_id{det_id}"
        p = outdir / f"{stem}{ext}"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths
