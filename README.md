# Widget Annotation Tool 

This work is part of our contributions in the paper [TargetFinder: Detecting Widget Information from Pixels on Desktop Interfaces](https://....). The objective is to detect GUI widgets in real-time using the YOLOv8 model, enabling smart interaction techniques. To achieve this, a good quality dataset is essential to retarin YOLOv8. Annotated datasets for desktop interfaces are much rarer than for mobile UIs. Therefore, we created this tool to facilitate the annotation process efficiently.

The tool supports pre-labeling based on existing methods: **REMAUI**, **UIED**, **Screen2SOM**, and **MobileSAM**. We also integrate our trained model as an additional technique.

To classify the detected components, we use three CNN-based classifiers:
  - One is the original UIED CNN trained on the RICO dataset (mobile interfaces),
  - Two others are from the SmartUI repo (another work during a competition based on UIED): one trained on wireframes, and the other on a combined dataset (wireframes + ReDraw). Link to the repo: https://github.com/tezansahu/smart_ui_tf20/tree/main

For more technical details and usage, see the Help menu or the Info buttons inside the application.

## Features
- Automatic pre-labeling of interface elements, saving time.
- User-friendly GUI to correct, adjust, or delete annotations: Zoom, selection, pixel-perfect editing (via keyboard keys), etc.
- Saving and loading annotations in YOLO format.

## Demo
![App Demo](assets/gif.gif)


## Project Structure

- `gui.py`                : Main PyQt application  
- `cnn_classifier.py`     : CNN classification logic  
- `technique1.py`         : REMAUI – Step 2  
- `technique2.py`         : Screen2SOM – M4  
- `technique3.py`         : MobileSAM  
- `technique4.py`         : UIED  
- `technique5.py`         : TargetFinder (our trained YOLOv8)  
- `method_docs.py`        : HTML documentation for all methods  
- `models/`               : Folder for trained models (MobileSAM, M4, CNNs)  
- `assets/`               : Logos and demo images for the GUI  
- `uiComponentDetector/`  : UIED modules  


## Scientific References

- **REMAUI**  
  Tuan Anh Nguyen, Christoph Csallner (2015) – *Reverse Engineering Mobile Application User Interfaces With REMAUI*  
  [Read the article](https://ieeexplore.ieee.org/document/7372013)

- **UIED**  
  M. Xie, S. Feng, Z. Xing, J. Chen, and C. Chen. – *UIED: A Hybrid Tool for GUI Element Detection*  
  [Read the article](https://dl.acm.org/doi/10.1145/3368089.3417940)

  **Resources:** UIED is tested using the work <a href="https://github.com/tezansahu/smart_ui_tf20" target="_blank">smart_ui_tf20</a>, which was developed for the <strong>Smart UI Competition</strong> (TechFest 2020-21).

- **Screen2SOM**  
  A. Martínez-Rojas, A. Rodríguez-Ruíz, J. G. Enríquez, and A. Jiménez-Ramírez – *What’s Behind the Screen? Unveiling UI Hierarchies in Process-Related UI Logs*  
  [Read the article](https://doi.org/10.1007/978-3-031-70396-6_15)

- **MobileSAM**  
  By [Ultralytics](https://ultralytics.com)  
  [Documentation](https://docs.ultralytics.com/models/mobile-sam/)

---



## Usage
Tested with Python 3.10 and 3.11. Below is an example of how to run the tool using Anaconda:

Clone the project :

```console
> git clone https://github.com/...
> cd ui-widget-annotator-a-tool-for-fast-and-efficient-gui-annotation/
```
> **Note:** You must first download and place the classification models in the "models" folder. Here is the link to download them. [link](https://drive.google.com/drive/folders/1Z374I1fQWqe1iykFkrkd-nCarlqAquot?usp=drive_link)

Create a conda environment:

```console
> conda create -n gui-annotator python=3.11
> conda activate gui-annotator
```

Install the dependencies:

```console
(gui-annotator)> pip install -r requirements.txt
```


Launch the app:

```console
(gui-annotator)> python gui.py
```

The startup of the PyQt interface may take a few seconds, as we import TensorFlow components before launching the PyQr interface to avoid DDL issues.

Then:
- Select a folder containing images.
- Choose a method and launch detection.
- Refine annotations if needed, classify components, and save the results.
