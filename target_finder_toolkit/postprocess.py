"""
Utility functions to consume the detection callback output of TargetFinder.

This module provides helpers for:
    - Selection, sorting, and filtering of detections.
    - Drawing and exporting annotated frames (bounding boxes, labels).
    - Extracting and saving crops of detected widgets.

Expected data structures:
    - detections: list of dicts with keys ``{id, x, y, width, height, score, class_id, class_name}``
    - frame: RGB image as a NumPy array.
"""

import json
from pathlib import Path
import numpy as np
import cv2


# ============== Selection / sorting / filtering ==============

def get_ids(detections):
    """Return sorted detection IDs (ascending).

    Args:
        detections (list[dict]): Detections, each dict may contain key ``"id"``.

    Returns:
        list[int]: Sorted IDs ``id``.
    """
    return sorted(int(d["id"]) for d in detections if d.get("id") is not None)


def sort_detections(detections, key="id", reverse=False):
    """Return a new list of detections sorted by a given key.

    Args:
        detections (list[dict]): Detection dictionaries.
        key (str, optional): Key to sort by (e.g., ``"id"``, ``"score"``,
            ``"x"``, ``"y"``, ``"width"``, ``"height"``). Defaults to ``"id"``.
        reverse (bool, optional): If ``True``, sort in descending order. Defaults to ``False``.

    Returns:
        list[dict]: Sorted copy of the input detections.
    """

    return sorted(detections, key=lambda d: d.get(key, 0), reverse=reverse)


def summarize_change(detections, added, removed):
    """Build a compact summary of changes between two detection sets.

    Args:
        detections (list[dict]): Current detections.
        added (list[dict]): Detections that appeared since the previous step.
        removed (list[dict]): Detections that disappeared since the previous step.

    Returns:
        dict: Summary dictionary with keys:
            - ``total`` (int): Number of detections.
            - ``added_ids`` (list[int]): IDs of newly added detections.
            - ``removed_ids`` (list[int]): IDs of removed detections.
            - ``current_ids`` (list[int]): IDs of current detections.
    """
    return {
        "total": len(detections),
        "added_ids": get_ids(added),
        "removed_ids": get_ids(removed),
        "current_ids": get_ids(detections),
    }


def pretty_print_change(detections, added, removed):
    """Print a change summary (useful in quick logs / tests).

    Args:
        detections (list[dict]): Current detections.
        added (list[dict]): Newly added detections.
        removed (list[dict]): Removed detections.
    """

    s = summarize_change(detections, added, removed)
    print(f"{s['total']} targets (+{len(s['added_ids'])} / -{len(s['removed_ids'])})")


# ============== Drawing / export ==============

def draw_bboxes(frame_rgb, detections, draw_ids=True, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes and labels on a copy of an RGB frame.

    The label format is ``"<class_name>:<score>[#<id>]"`` when ``draw_ids=True``.

    Args:
        frame_rgb (np.ndarray): HxWx3 RGB image.
        detections (list[dict]): Detection dicts with keys
            ``x``, ``y``, ``width``, ``height`` (and optionally ``class_name``,
            ``score``, ``id``).
        draw_ids (bool, optional): If ``True``, append ``#<id>`` to the label
            when present. Defaults to ``True``.
        color (tuple[int, int, int], optional): Box color in BGR.
            Defaults to ``(0, 255, 0)``.
        thickness (int, optional): Line thickness. Defaults to ``2``.

    Returns:
        np.ndarray: Annotated RGB image.
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
    """Save an annotated PNG image.

    Args:
        path (str | Path): Output file path.
        frame_rgb (np.ndarray): HxWx3 RGB image.
        detections (list[dict]): Detections to draw.
        draw_ids (bool, optional): If ``True``, include IDs in labels when available.

    Returns:
        Path: Path to the saved image.
    """
    out = draw_bboxes(frame_rgb, detections, draw_ids=draw_ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)
    return path


def save_detections_json(path, detections):
    """Save detections to a JSON file.

    Args:
        path (str | Path): Output JSON path.
        detections (list[dict]): Detections to serialize.

    Returns:
        Path: Path to the saved JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)
    return path


# ============== Crops ==============

def extract_crops(frame_rgb, detections, clamp=True):
    """Extract cropped widget images from a screen frame. This utility takes the full RGB screenshot and a list of detections,
    then returns individual image crops (sub-images) corresponding to each
    detected GUI widget (e.g., button, slider, text field).

    Args:
        frame_rgb (np.ndarray): HxWx3 RGB image.
        detections (list[dict]): Detection dicts with keys ``x``, ``y``,
            ``width``, ``height`` (and optionally ``id``).
        clamp (bool, optional): If ``True``, crop boxes are clamped to
            image bounds. Defaults to ``True``.

    Returns:
        list[tuple[int, np.ndarray]]: List of ``(det_id, crop)``, where
        ``det_id`` is the widget ID.
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
    """Save a list of crops to a directory.

    Filenames follow the pattern:
        ``<prefix>_<index>[_id<ID>]<ext>``

    **Examples:** ``crop_0000.png`` - ``crop_0001_id12.png``

    Args:
        outdir (str | Path): Output directory.
        crops (list[tuple[int, np.ndarray]]): List of ``(det_id, crop)`` pairs
            as returned by :func:`extract_crops`.
        prefix (str, optional): Filename prefix. Defaults to ``"crop"``.
        ext (str, optional): File extension (e.g., ``".png"``). Defaults to ``".png"``.

    Returns:
        list[Path]: Paths to the saved crops.
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
