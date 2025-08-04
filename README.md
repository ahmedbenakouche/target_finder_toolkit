# Target Finder Toolkit

Toolkit for target detection and interaction using techniques like **Bubble Cursor** and **Semantic Pointing**.  
Provides CLI/GUI entry points to experiment with and visualize these interaction methods.

## Demo (visual preview)

### Animated previews

Click on each GIF below to open its corresponding full video (`.mp4` with the same base name).

#### Linux
[![Target Finder - Linux](./demo/GIFs/linux_TargetFinder.gif)](./demo/Videos/linux_TargetFinder.mp4)  
[![Bubble Cursor - Linux](./demo/GIFs/linux_bubble_cursor.gif)](./demo/Videos/linux_bubble_cursor.mp4)  
[![Semantic Pointing - Linux](./demo/GIFs/linux_semantic_pointing.gif)](./demo/Videos/linux_semantic_pointing.mp4)  

#### Windows
[![Target Finder - Windows](./demo/GIFs/windows_TargetFinder.gif)](./demo/Videos/windows_TargetFinder.mp4)  
[![Bubble Cursor - Windows](./demo/GIFs/windows_bubble_cursor.gif)](./demo/Videos/windows_bubble_cursor.mp4)  
[![Semantic Pointing - Windows](./demo/GIFs/windows_semantic_pointing.gif)](./demo/Videos/windows_semantic_pointing.mp4)  

> If storing full videos in the repository is undesirable due to size, you can publish them as a **GitHub Release** and replace the above links with the release asset URLs.

## Installation (development)

```bash
git clone https://github.com/AHMEDBENAKOUCHE/target_finder_toolkit.git
cd target_finder_toolkit

# create and activate a virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Unix/macOS
# source .venv/bin/activate

# install the package (and its dependencies) in editable mode
pip install -e .
