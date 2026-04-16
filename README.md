# TargetFinder Toolkit

This toolkit accompanies the work presented in the article **TargetFinder: Detecting Widgets from Pixels on Desktop Interfaces**.  
It provides a real-time detection system using the YOLO26 model to predict the bounding boxes of GUI widgets from desktop screenshots — **without requiring access to application internals or accessibility APIs**.

The system is lightweight and easy to integrate, enabling the implementation of advanced interaction techniques.  
As proof of concept, we include two interaction techniques built on top of TargetFinder:

- **[Bubble Cursor](https://dl.acm.org/doi/10.1145/1054972.1055012)** 
- **[Semantic Pointing](https://dl.acm.org/doi/10.1145/985692.985758)** 

> **Compatibility Note**  
> TargetFinder uses the `mss` library for fast screen capture, and the detection engine is theoretically cross-platform. However, the system has been **validated only on Windows 10/11 and Linux (Ubuntu X11)**.
> Operation is **not guaranteed on macOS**, where additional adaptations are required.  
> Other Linux setups (e.g., distributions other than Ubuntu, or Wayland instead of X11) may also require adjustments.


---

## Installation

```bash
pip install .
```

<details>
<summary><strong>Linux prerequisites (click to expand)</strong></summary>

During installation on Linux, you may need to install some system packages to avoid common errors:

1. **evdev build tools**  
   If installation fails due to missing `evdev` headers:  
   ```bash
   sudo apt install build-essential python3-dev
   ```

2. **X11 vs Wayland screen capture**  
   `mss` relies on X11 (does not work with Wayland since `XGetImage` is not available).  
   If you see the following error:  
   ```
   mss.exception.ScreenShotError: XGetImage() failed
   ```
   Switch to an X11 session at login.

3. **Qt X11 plugin (`xcb`)**  
   If you encounter errors like:  
   ```
   qt.qpa.plugin: Could not load the Qt platform plugin "xcb" ...
   ```
   Install the required libraries:  
   ```bash
   sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0
   ```

4. **tk (MouseInfo) support**  
   `pyautogui` or `pynput` may fail if `tk` is missing:  
   ```bash
   sudo apt install python3-tk python3-dev
   ```

</details>

### Python API & Examples

Minimal examples:

#### Print detection changes (callback)

```python
import time
from target_finder_toolkit.targetfinder import TargetFinder
from target_finder_toolkit import postprocess as pp

def on_change(detections, added, removed, frame):
    """
    Called every time the detector refreshes.
    - detections: [{id, x, y, width, height, score, class_id, class_name} ...]
    - added: new detections since last callback
    - removed: detections that disappeared since last callback
    - frame: RGB numpy array (only if with_frame=True), else None
    """
    pp.pretty_print_change(detections, added, removed)

if __name__ == "__main__":
    det = TargetFinder()
    det.set_callback(on_change, with_frame=False, diff_iou=0.5)
    det.start()

    print("TargetFinder started — press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        det.stop()
        print("Stopped.")
```

#### Save one annotated frame + crops

```python
import time
from pathlib import Path
from target_finder_toolkit.targetfinder import TargetFinder
from target_finder_toolkit import postprocess as pp

# global flag to ensure we only save once
_saved_once = False

def on_change(detections, added, removed, frame):
    """
    Save a single annotated frame and crops (first time we get detections+frame).
    """
    global _saved_once
    if _saved_once:
        return
    if frame is None or not detections:
        return

    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save annotated frame
    annot_path = out_dir / "annotated.png"
    pp.save_annotated(annot_path, frame, detections)

    # Extract and save crops
    crops = pp.extract_crops(frame, detections)
    crop_paths = pp.save_crops(out_dir / "crops", crops, prefix="crop", ext=".png")

    print(f"Saved annotated frame: {annot_path}")
    print(f"Saved {len(crop_paths)} crops under: {out_dir/'crops'}")

    _saved_once = True

if __name__ == "__main__":
    det = TargetFinder()
    # with_frame=True is required to receive the RGB frame in the callback
    det.set_callback(on_change, with_frame=True, diff_iou=0.5)
    det.start()

    print("TargetFinder started — will save once when detections are available. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        det.stop()
        print("Stopped.")

```

#### Detect on a static image

```python
from target_finder_toolkit.targetfinder import TargetFinder

if __name__ == "__main__":
    image_path = "screenshot.png"  # change this to your image

    det = TargetFinder()
    detections = det.detect_image(
        image_path,
        save_annotated=True,   # creates e.g. screenshot_annotated.png
        save_json=True         # creates e.g. screenshot_detections.json
    )
```

## Documentation

For the full API reference and detailed explanations of all parameters,  
visit the documentation site:

👉 [**Documentation (API & Developer Guide)**](https://target-finder-toolkit.netlify.app/)



## Demos using TargetFinder

### TargetFinder GUI

After installation, `targetfinder-gui`: launches the main overlay GUI.

| Windows | Linux |
|--------|--------|
| ![Windows Target Finder](./demo/GIFs/windows_TargetFinder.gif) | ![Linux Target Finder](./demo/GIFs/linux_TargetFinder.gif) |
| **Full video: (see in /demo/Videos/)** | **Full video: (see in /demo/Videos/)** |

### Bubble Cursor

After installation, `bubblecursor` runs the Bubble Cursor interaction technique.

| Windows | Linux |
|--------|--------|
| ![Bubble Cursor - Windows](./demo/GIFs/windows_bubble_cursor.gif) | ![Bubble Cursor - Linux](./demo/GIFs/linux_bubble_cursor.gif) |
| **Full video: (see in /demo/Videos/)** | **Full video: (see in /demo/Videos/)** |

### Semantic Pointing

After installation, `semanticpointing` runs the Semantic Pointing interaction technique.

| Windows | Linux |
|--------|--------|
| ![Semantic Pointing - Windows](./demo/GIFs/windows_semantic_pointing.gif) | ![Semantic Pointing - Linux](./demo/GIFs/linux_semantic_pointing.gif) |
| **Full video: (see in /demo/Videos/)** | **Full video: (see in /demo/Videos/)** |


#### Available options:

| Option | Description |
|--------|-------------|
| `--model-path` | By default, TargetFinder loads our trained model `YOLO26` packaged with the toolkit, but you can supply your own. |
| `--change-thresh` | Screen change detection threshold. A higher value makes detection less sensitive to small variations. (`default = 100`). |
| `--capture-interval` | Time between screen captures in seconds. Lower values = higher frequency but more CPU/GPU usage. (`default = 1/30 ≈ 0.033s`). |
| `--confidence` | Minimum YOLO confidence required to keep a detection. (`[0.0–1.0], default = 0.28`). |
| `--iou` | IoU threshold for non-max suppression (controls overlap merging). (`[0.0–1.0], default = 0.3`). |
| `--display` *(semanticpointing only)* | Show visual feedback (motor vs visual space). |
| `--disable-accel` *(semanticpointing only)* | Disable system mouse acceleration. |

Example:
```bash
bubblecursor \
  --change-thresh 100 \
  --capture-interval 0.033 \
  --confidence 0.28 \
  --iou 0.3
```

---
