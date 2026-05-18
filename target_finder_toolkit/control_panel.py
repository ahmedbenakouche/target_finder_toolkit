import json
import importlib
import os
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from PyQt6 import QtCore, QtWidgets
from target_finder_toolkit.filters import (
    DEFAULT_FILTER_BETA,
    DEFAULT_FILTER_D_CUTOFF,
    DEFAULT_FILTER_FREQ,
    DEFAULT_FILTER_MIN_CUTOFF,
    FILTER_OPTIONS,
)
from target_finder_toolkit.logging_utils import make_default_log_path
from target_finder_toolkit.mouse_utils import restore_default_cursors


def _ensure_mediapipe_python_alias():
    """Provide a compatibility alias for WebEyeTrack on newer MediaPipe builds."""
    try:
        import mediapipe as mp
    except Exception:
        return
    if "mediapipe.python" not in sys.modules:
        sys.modules["mediapipe.python"] = mp
    if not hasattr(mp, "python"):
        mp.python = mp


MODE_OPTIONS = {
    "targetfinder": {
        "English": "TargetFinder Overlay",
        "French": "Overlay TargetFinder",
    },
    "bubble": {
        "English": "Bubble Cursor",
        "French": "Bubble Cursor",
    },
    "semantic": {
        "English": "Semantic Pointing",
        "French": "Pointage sémantique",
    },
    "dynaspot": {
        "English": "DynaSpot",
        "French": "DynaSpot",
    },
    "rake": {
        "English": "Ninja Cursors(gaze)",
        "French": "Ninja Cursors(gaze)",
    },
}

LANGUAGE_OPTIONS = {
    "English": {
        "English": "English",
        "French": "Anglais",
        "badge": "🇬🇧",
    },
    "French": {
        "English": "French",
        "French": "Français",
        "badge": "🇫🇷",
    },
}

DEFAULT_CHANGE_THRESH = 100
DEFAULT_CAPTURE_INTERVAL = 0.033
DEFAULT_CONFIDENCE = 0.28
DEFAULT_IOU = 0.3
DEFAULT_DYNASPOT_MIN_SPEED = 100.0
DEFAULT_DYNASPOT_SPOT_WIDTH = 32.0
DEFAULT_DYNASPOT_LAG = 0.12
DEFAULT_DYNASPOT_REDUCE_TIME = 0.18
DEFAULT_RAKE_CAMERA_INDEX = 0
DEFAULT_RAKE_SCREEN_WIDTH_CM = 34.0
DEFAULT_RAKE_SCREEN_HEIGHT_CM = 19.0
DEFAULT_RAKE_SPACING = 320.0
DEFAULT_RAKE_GAZE_SMOOTHING = 0.35
DEFAULT_RAKE_GAZE_GAIN_X = 1.0
DEFAULT_RAKE_GAZE_GAIN_Y = 1.0
DEFAULT_RAKE_GAZE_OFFSET_X = 0.0
DEFAULT_RAKE_GAZE_OFFSET_Y = -200.0
DEFAULT_RAKE_SELECTION_HOLD = 2.0
DEFAULT_RAKE_LOCK_ON_DWELL = False
DEFAULT_RAKE_USE_CALIBRATION = True
DEFAULT_RAKE_CALIB_POINTS = 5
DEFAULT_RAKE_AUTO_CALIBRATE = False
DEFAULT_RAKE_WITHOUT_TARGETFINDER = True
DEFAULT_EXPERIMENT_DATA_DIR = str(Path(__file__).resolve().parents[3] / "data" / "web")
DEFAULT_EXPERIMENT_TRIALS = 12
DEFAULT_EXPERIMENT_DIFFICULTY = "mixed"
DEFAULT_EXPERIMENT_COUNTDOWN = 3
DEFAULT_EXPERIMENT_MAX_CLICKS = 1
DEFAULT_EXPERIMENT_FULLSCREEN = True
DEFAULT_EXPERIMENT_SHOW_ALL_TARGETS = False

UI_TEXTS = {
    "English": {
        "nav_mode": "Mode / Detection",
        "nav_accessibility": "Accessibility",
        "nav_audio": "Audio",
        "nav_language": "Language",
        "page_mode": "Mode / Detection",
        "page_accessibility": "Accessibility",
        "page_audio": "Audio",
        "page_language": "Language",
        "setup_section": "Technique setup",
        "filter_section": "Pointer filter",
        "logging_section": "Logging",
        "detection_section": "Detection refresh",
        "experiment_options_section": "Experimental task options",
        "semantic_section": "Semantic Pointing options",
        "dynaspot_section": "DynaSpot options",
        "rake_section": "Ninja Cursors(gaze) options",
        "technique": "Technique (5 modes, default: none)",
        "select_technique": "Choose a technique",
        "choose_mode_dialog": "Choose a Technique",
        "filter": "Filter (choices: none/one euro, default: none)",
        "select_filter": "Choose a filter",
        "choose_filter_dialog": "Choose a Filter",
        "filter_none": "None",
        "filter_one_euro": "One Euro",
        "filter_desc": "Apply an optional cursor filter before the selected technique uses pointer input.",
        "filter_params": "One Euro filter parameters",
        "filter_params_desc": "Advanced parameters for the One Euro filter. Usually fixed during demos; tune only if the defaults are not good enough.",
        "filter_freq": "Filter frequency (range: 1.0-1000.0, default: 120.0)",
        "filter_freq_desc": "Expected sampling frequency used by the One Euro filter.",
        "filter_min_cutoff": "Filter min cutoff (range: 0.001-100.0, default: 1.0)",
        "filter_min_cutoff_desc": "Lower values smooth more at low speed; higher values react faster but can jitter more.",
        "filter_beta": "Filter beta (range: 0.0-10.0, default: 0.02)",
        "filter_beta_desc": "Speed adaptation factor. Higher values reduce lag during fast movement.",
        "filter_d_cutoff": "Filter derivative cutoff (range: 0.001-100.0, default: 1.0)",
        "filter_d_cutoff_desc": "Smoothing cutoff for the speed estimate used by the filter.",
        "record_data": "Record session data (range: off/on, default: off)",
        "record_data_desc": "Save structured JSONL logs with cursor samples, clicks, and detection changes for later analysis.",
        "mode_targetfinder": "TargetFinder Overlay",
        "mode_bubble": "Bubble Cursor",
        "mode_semantic": "Semantic Pointing",
        "mode_dynaspot": "DynaSpot",
        "mode_rake": "Ninja Cursors(gaze)",
        "dynaspot_params": "DynaSpot tuning",
        "dynaspot_min_speed": "DynaSpot min speed (range: 0.0-5000.0, default: 100.0)",
        "dynaspot_min_speed_desc": "Pointer speed threshold where the spot starts growing. Lower values make the spot expand earlier.",
        "dynaspot_spot_width": "DynaSpot spot width (range: 1.0-128.0, default: 32.0)",
        "dynaspot_spot_width_desc": "Maximum activation area width in pixels. This is the paper's SPOTWIDTH and refers to diameter, not radius.",
        "dynaspot_lag": "DynaSpot shrink lag (range: 0.0-5.0, default: 0.12)",
        "dynaspot_lag_desc": "Delay before the spot starts shrinking once the pointer stops moving.",
        "dynaspot_reduce_time": "DynaSpot reduce time (range: 0.001-10.0, default: 0.18)",
        "dynaspot_reduce_time_desc": "Time used for the co-exponential reduction back toward a 1-pixel point cursor.",
        "rake_params": "Ninja Cursors(gaze) tuning",
        "rake_camera_index": "Webcam index (range: 0-10, default: 0)",
        "rake_camera_index_desc": "Camera index used by WebEyeTrack for webcam-based gaze estimation.",
        "rake_screen_width_cm": "Screen width (cm) (range: 10.0-200.0, default: auto-detected current screen)",
        "rake_screen_width_cm_desc": "Auto-detected physical screen width used by WebEyeTrack. You can override it if detection is wrong.",
        "rake_screen_height_cm": "Screen height (cm) (range: 10.0-200.0, default: auto-detected current screen)",
        "rake_screen_height_cm_desc": "Auto-detected physical screen height used by WebEyeTrack. You can override it if detection is wrong.",
        "rake_spacing": "Ninja spacing (range: 80.0-800.0, default: 320.0)",
        "rake_spacing_desc": "Controls how widely the 8 cursors are spread. Lower = cursors closer together. Higher = cursors farther apart. The default reproduces the paper-style 4x2 layout.",
        "rake_gaze_smoothing": "Gaze smoothing (range: 0.0-0.95, default: 0.35)",
        "rake_gaze_smoothing_desc": "Per frame, the system keeps this fraction of the previous gaze point and uses the rest from the new webcam sample. Higher = steadier but more lag.",
        "rake_calibration_section": "Calibration",
        "rake_use_calibration": "Use calibration (range: off/on, default: on)",
        "rake_use_calibration_desc": "Runs gaze calibration before Ninja Cursors starts. The resulting correction values fill the editable gain/offset fields.",
        "rake_calibration_points": "Calibration points (choices: 5/9/13, default: 5)",
        "rake_calibration_points_desc": "Number of points used during multi-point calibration. More points usually improve accuracy but take longer.",
        "rake_calibration_actions": "Calibration actions",
        "rake_calibration_actions_desc": "When calibration is enabled, Start / Apply launches Ninja Cursors(gaze), begins calibration automatically, and fills the correction fields when done.",
        "rake_calibration_status": "Calibration status",
        "rake_calibration_status_desc": "Current calibration state used by the panel.",
        "rake_calibration_status_not_calibrated": "Not calibrated",
        "rake_calibration_status_calibrating": "Calibrating...",
        "rake_calibration_status_calibrated": "Calibrated",
        "rake_calibration_status_failed": "Calibration failed",
        "rake_calibration_status_cancelled": "Calibration cancelled",
        "rake_calibration_status_last_applied": "Last calibration applied",
        "rake_calibration_mode": "Correction mode",
        "rake_calibration_mode_desc": "Shows whether Ninja Cursors(gaze) is currently using manual correction or calibration mode.",
        "rake_calibration_mode_manual": "Manual correction mode",
        "rake_calibration_mode_active": "Calibration mode",
        "rake_calibration_manual_enabled": "Manual correction mode: gaze offset and gain remain editable.",
        "rake_calibration_manual_disabled": "Calibration active: gain and offset will be filled automatically, then remain editable for manual tuning.",
        "rake_manual_disabled_hint": "Auto-filled after calibration; still editable",
        "rake_reset_calibration": "Reset calibration",
        "rake_gaze_gain_x": "Gaze gain X (range: 0.1-10.0, default: 1.0)",
        "rake_gaze_gain_x_desc": "Scales horizontal gaze movement around the screen center before offset is applied. Higher values increase left-right travel.",
        "rake_gaze_gain_y": "Gaze gain Y (range: 0.1-10.0, default: 1.0)",
        "rake_gaze_gain_y_desc": "Scales vertical gaze movement around the screen center before offset is applied. Higher values increase up-down travel.",
        "rake_gaze_offset_x": "Gaze offset X (px) (range: -1000.0-1000.0, default: 10.0)",
        "rake_gaze_offset_x_desc": "Shifts the gaze estimate horizontally before selecting the active cursor. Positive = move right, negative = move left.",
        "rake_gaze_offset_y": "Gaze offset Y (px) (range: -1000.0-1000.0, default: -200.0)",
        "rake_gaze_offset_y_desc": "Shifts the gaze estimate vertically before selecting the active cursor. Positive = move down, negative = move up.",
        "rake_lock_on_dwell": "Lock cursor by gaze dwell (range: off/on, default: off)",
        "rake_lock_on_dwell_desc": "When enabled, gaze must stay on the same cursor before it locks. When disabled, the current yellow cursor can be clicked immediately.",
        "rake_selection_hold": "Gaze dwell lock time (range: 0.0-5.0, default: 2.0)",
        "rake_selection_hold_desc": "Seconds the gaze must stay on the same cursor before it locks automatically. Used only when dwell locking is enabled.",
        "rake_show_gaze": "Show gaze point (Ninja only, range: off/on, default: on)",
        "rake_show_gaze_desc": "Shows a red gaze marker estimated from the webcam.",
        "rake_without_targetfinder": "Without TargetFinder (range: off/on, default: on)",
        "rake_without_targetfinder_desc": "Runs Ninja Cursors(gaze) without detection, target highlighting, or model inference. Only gaze-based cursor selection and redirected clicks remain active.",
        "experiment_section": "Experimental task",
        "experiment_enabled": "Run experimental task (range: off/on, default: off)",
        "experiment_enabled_desc": "When enabled, Start / Apply launches the controlled screenshot target-selection task instead of a free demo.",
        "experiment_data_dir": "Dataset folder (default: stage/data/web)",
        "experiment_data_dir_desc": "Folder containing the annotated screenshot .png/.txt pairs used to generate trials.",
        "experiment_trials": "Trials (range: 1-1000, default: 12)",
        "experiment_trials_desc": "Number of controlled target-selection trials to run.",
        "experiment_difficulty": "Difficulty (choices: easy/medium/hard/mixed, default: mixed)",
        "experiment_difficulty_desc": "Difficulty bin sampled by Fitts ID: easy [0,3), medium [3,5), hard [5,8.5). Mixed samples from all bins.",
        "experiment_countdown": "Countdown (seconds, range: 0-30, default: 3)",
        "experiment_countdown_desc": "Seconds before each trial starts while the cursor is held at the image center.",
        "experiment_max_clicks": "Max clicks per trial (range: 1-20, default: 1)",
        "experiment_max_clicks_desc": "Maximum attempts allowed before a trial is marked failed.",
        "experiment_fullscreen": "Fullscreen experiment (range: off/on, default: on)",
        "experiment_fullscreen_desc": "Show the screenshot task fullscreen. Disable only for debugging.",
        "experiment_show_all_targets": "Show all annotated targets (debug, default: off)",
        "experiment_show_all_targets_desc": "Draw all dataset annotation boxes in green. Use only for debugging, not participant runs.",
        "experiment_note": "For the TargetFinder condition, the controlled task uses the annotated dataset as ground truth and runs the mouse baseline internally.",
        "apply": "Start / Apply",
        "change_thresh": "Change Threshold (range: 0-100000, default: 100)",
        "change_thresh_desc": "Higher = fewer refreshes for small screen changes. Lower = reacts sooner.",
        "model_path": "YOLO Model (.pt file) (default: packaged yolo26s_1280.pt)",
        "model_path_desc": "Leave empty to use the packaged yolo26s_1280.pt. Choose another trained .pt file to switch models.",
        "browse": "Browse",
        "use_default_model": "Use packaged yolo26s_1280.pt",
        "capture_interval": "Capture Interval (range: 0.001-10.0, default: 0.033)",
        "capture_interval_desc": "Lower = faster checks and more CPU/GPU use. Higher = slower updates.",
        "confidence": "Confidence (range: 0.0-1.0, default: 0.28)",
        "confidence_desc": "Lower = keeps more detections. Higher = keeps only more certain detections.",
        "iou": "IoU (range: 0.0-1.0, default: 0.3)",
        "iou_desc": "Lower = keeps more overlapping boxes. Higher = merges overlaps more aggressively.",
        "display": "Display visual feedback (semantic only, range: off/on, default: off)",
        "display_short": "Display visual feedback",
        "display_desc": "Shows semantic-pointing visual guides on screen when enabled.",
        "disable_accel": "Disable system mouse acceleration (semantic only, range: off/on, default: off)",
        "disable_accel_short": "Disable system mouse acceleration",
        "disable_accel_desc": "Makes semantic pointing feel more stable, but changes mouse behavior while running.",
        "mode_note": "TargetFinder Overlay: shows detected boxes for testing. Bubble Cursor: expands selection around the nearest target. Semantic Pointing: slows pointer movement near targets for easier aiming. DynaSpot: keeps the normal system cursor as the center and grows a circular activation area with speed while preserving empty-space clicks. Ninja Cursors(gaze): gaze first activates the nearest cursor among 8 distributed cursors; if the gaze stays there long enough, that cursor locks automatically for local refinement until the click finishes.",
        "contrast": "Contrast",
        "enable_tts": "Enable TTS",
        "language": "Language",
        "choose_language_dialog": "Choose a Language",
        "confirm": "Confirm",
        "cancel": "Cancel",
        "enter_value_for": "Enter a value for {name}.",
        "turn_on": "Turn on {name}",
        "turn_off": "Turn off {name}",
        "stop": "Stop Running Mode",
        "ready": "Ready. Choose settings, then press Start / Apply.",
        "pending_apply": "Settings updated. Press Start / Apply to launch or refresh the selected technique.",
        "select_mode_first": "Choose a technique first, then press Start / Apply.",
        "running_bubble": "Bubble Cursor is running.",
        "running_semantic": "Semantic Pointing is running.",
        "running_targetfinder": "TargetFinder Overlay is running.",
        "running_dynaspot": "DynaSpot is running.",
        "running_rake": "Ninja Cursors(gaze) is running.",
        "running_experiment": "Experimental task is running.",
        "stopped": "Stopped the running mode.",
        "no_running": "No running mode was found.",
        "invalid_model_path": "The selected model file was not found.",
        "invalid_experiment_data_dir": "The selected experimental dataset folder was not found.",
        "missing_one_euro": "OneEuroFilter is not installed or could not be imported in this environment.",
        "missing_webeyetrack": "WebEyeTrack is not installed or could not be imported in this environment.",
        "panel_updated": "Panel appearance updated.",
        "tts_enabled": "Text-to-speech enabled.",
        "tts_disabled": "Text-to-speech disabled.",
        "tts_unavailable": "Text-to-speech is not available on this system.",
        "language_updated": "Interface language updated.",
        "q_hint": "You can also press q to quit the running mode.",
    },
    "French": {
        "nav_mode": "Mode / Detection",
        "nav_accessibility": "Accessibilité",
        "nav_audio": "Audio",
        "nav_language": "Langue",
        "page_mode": "Mode / Detection",
        "page_accessibility": "Accessibilité",
        "page_audio": "Audio",
        "page_language": "Langue",
        "setup_section": "Configuration de la technique",
        "filter_section": "Filtre du pointeur",
        "logging_section": "Enregistrement",
        "detection_section": "Détection et rafraîchissement",
        "experiment_options_section": "Options de la tâche expérimentale",
        "semantic_section": "Options du pointage sémantique",
        "dynaspot_section": "Options de DynaSpot",
        "rake_section": "Options de Ninja Cursors(gaze)",
        "technique": "Technique (5 modes, défaut : aucun)",
        "select_technique": "Choisir une technique",
        "choose_mode_dialog": "Choisir une technique",
        "filter": "Filtre (choix : none/one euro, défaut : none)",
        "select_filter": "Choisir un filtre",
        "choose_filter_dialog": "Choisir un filtre",
        "filter_none": "Aucun",
        "filter_one_euro": "One Euro",
        "filter_desc": "Appliquer un filtre optionnel au pointeur avant que la technique sélectionnée n'utilise l'entrée souris.",
        "filter_params": "Paramètres du filtre One Euro",
        "filter_params_desc": "Paramètres avancés du filtre One Euro. En démonstration, ils restent généralement fixes ; à modifier seulement si les valeurs par défaut ne conviennent pas.",
        "filter_freq": "Fréquence du filtre (plage : 1.0-1000.0, défaut : 120.0)",
        "filter_freq_desc": "Fréquence d'échantillonnage attendue par le filtre One Euro.",
        "filter_min_cutoff": "Coupure minimale du filtre (plage : 0.001-100.0, défaut : 1.0)",
        "filter_min_cutoff_desc": "Plus bas = plus lisse à basse vitesse ; plus haut = plus réactif mais potentiellement plus instable.",
        "filter_beta": "Bêta du filtre (plage : 0.0-10.0, défaut : 0.02)",
        "filter_beta_desc": "Facteur d'adaptation à la vitesse. Plus haut réduit le retard lors des mouvements rapides.",
        "filter_d_cutoff": "Coupure dérivée du filtre (plage : 0.001-100.0, défaut : 1.0)",
        "filter_d_cutoff_desc": "Coupure de lissage pour l'estimation de vitesse utilisée par le filtre.",
        "record_data": "Enregistrer les données (plage : off/on, défaut : off)",
        "record_data_desc": "Enregistrer des journaux JSONL structurés avec la trajectoire du pointeur, les clics et les changements de détection.",
        "mode_targetfinder": "Overlay TargetFinder",
        "mode_bubble": "Bubble Cursor",
        "mode_semantic": "Pointage sémantique",
        "mode_dynaspot": "DynaSpot",
        "mode_rake": "Ninja Cursors(gaze)",
        "dynaspot_params": "Réglages DynaSpot",
        "dynaspot_min_speed": "Vitesse min DynaSpot (plage : 0.0-5000.0, défaut : 100.0)",
        "dynaspot_min_speed_desc": "Seuil de vitesse à partir duquel le spot commence à grandir. Plus bas = expansion plus précoce.",
        "dynaspot_spot_width": "Largeur du spot DynaSpot (plage : 1.0-128.0, défaut : 32.0)",
        "dynaspot_spot_width_desc": "Largeur maximale de la zone d’activation en pixels. C’est le SPOTWIDTH de l’article et il s’agit du diamètre, pas du rayon.",
        "dynaspot_lag": "Délai de réduction DynaSpot (plage : 0.0-5.0, défaut : 0.12)",
        "dynaspot_lag_desc": "Temps d’attente avant que le spot commence à diminuer lorsque le pointeur s’arrête.",
        "dynaspot_reduce_time": "Temps de réduction DynaSpot (plage : 0.001-10.0, défaut : 0.18)",
        "dynaspot_reduce_time_desc": "Durée de la réduction co-exponentielle pour revenir vers un curseur ponctuel de 1 pixel.",
        "rake_params": "Réglages Ninja Cursors(gaze)",
        "rake_camera_index": "Index de webcam (plage : 0-10, défaut : 0)",
        "rake_camera_index_desc": "Index de la caméra utilisée par WebEyeTrack pour estimer le regard.",
        "rake_screen_width_cm": "Largeur écran (cm) (plage : 10.0-200.0, défaut : écran courant détecté automatiquement)",
        "rake_screen_width_cm_desc": "Largeur physique de l’écran détectée automatiquement et utilisée par WebEyeTrack. Vous pouvez la corriger si la détection est incorrecte.",
        "rake_screen_height_cm": "Hauteur écran (cm) (plage : 10.0-200.0, défaut : écran courant détecté automatiquement)",
        "rake_screen_height_cm_desc": "Hauteur physique de l’écran détectée automatiquement et utilisée par WebEyeTrack. Vous pouvez la corriger si la détection est incorrecte.",
        "rake_spacing": "Espacement Ninja (plage : 80.0-800.0, défaut : 320.0)",
        "rake_spacing_desc": "Contrôle à quel point les 8 curseurs sont espacés. Plus bas = plus rapprochés. Plus haut = plus éloignés. La valeur par défaut reproduit la disposition 4x2 de l’article.",
        "rake_gaze_smoothing": "Lissage du regard (plage : 0.0-0.95, défaut : 0.35)",
        "rake_gaze_smoothing_desc": "À chaque frame, le système garde cette fraction de l’ancien point de regard et prend le reste depuis la nouvelle mesure webcam. Plus haut = plus stable mais plus de retard.",
        "rake_calibration_section": "Calibration",
        "rake_use_calibration": "Utiliser la calibration (plage : off/on, défaut : on)",
        "rake_use_calibration_desc": "Lance une calibration du regard avant Ninja Cursors. Les corrections obtenues remplissent les champs gain/décalage, qui restent modifiables.",
        "rake_calibration_points": "Points de calibration (choix : 5/9/13, défaut : 5)",
        "rake_calibration_points_desc": "Nombre de points utilisés pendant la calibration multipoint. Davantage de points améliore souvent la précision mais prend plus de temps.",
        "rake_calibration_actions": "Actions de calibration",
        "rake_calibration_actions_desc": "Quand la calibration est activée, Démarrer / Appliquer lance Ninja Cursors(gaze), démarre automatiquement la calibration et remplit les champs de correction.",
        "rake_calibration_status": "État de calibration",
        "rake_calibration_status_desc": "État actuel de calibration utilisé par le panneau.",
        "rake_calibration_status_not_calibrated": "Non calibré",
        "rake_calibration_status_calibrating": "Calibration en cours...",
        "rake_calibration_status_calibrated": "Calibré",
        "rake_calibration_status_failed": "Calibration échouée",
        "rake_calibration_status_cancelled": "Calibration annulée",
        "rake_calibration_status_last_applied": "Dernière calibration appliquée",
        "rake_calibration_mode": "Mode de correction",
        "rake_calibration_mode_desc": "Indique si Ninja Cursors(gaze) utilise actuellement le mode manuel ou le mode calibration.",
        "rake_calibration_mode_manual": "Mode de correction manuelle",
        "rake_calibration_mode_active": "Mode calibration",
        "rake_calibration_manual_enabled": "Mode de correction manuelle : le gain et le décalage du regard restent modifiables.",
        "rake_calibration_manual_disabled": "Calibration active : le gain et le décalage seront remplis automatiquement puis restent modifiables.",
        "rake_manual_disabled_hint": "Rempli automatiquement après calibration ; reste modifiable",
        "rake_reset_calibration": "Réinitialiser la calibration",
        "rake_gaze_gain_x": "Gain du regard X (plage : 0.1-10.0, défaut : 1.0)",
        "rake_gaze_gain_x_desc": "Agrandit ou réduit l’amplitude horizontale du regard autour du centre de l’écran avant d’appliquer le décalage. Plus haut = plus de déplacement gauche-droite.",
        "rake_gaze_gain_y": "Gain du regard Y (plage : 0.1-10.0, défaut : 1.0)",
        "rake_gaze_gain_y_desc": "Agrandit ou réduit l’amplitude verticale du regard autour du centre de l’écran avant d’appliquer le décalage. Plus haut = plus de déplacement haut-bas.",
        "rake_gaze_offset_x": "Décalage regard X (px) (plage : -1000.0-1000.0, défaut : 10.0)",
        "rake_gaze_offset_x_desc": "Décale l’estimation du regard horizontalement avant de choisir le curseur actif. Positif = vers la droite, négatif = vers la gauche.",
        "rake_gaze_offset_y": "Décalage regard Y (px) (plage : -1000.0-1000.0, défaut : -200.0)",
        "rake_gaze_offset_y_desc": "Décale l’estimation du regard verticalement avant de choisir le curseur actif. Positif = vers le bas, négatif = vers le haut.",
        "rake_lock_on_dwell": "Verrouiller par fixation du regard (plage : off/on, défaut : off)",
        "rake_lock_on_dwell_desc": "Si activé, le regard doit rester sur le même curseur avant verrouillage. Sinon, le curseur jaune courant peut être cliqué immédiatement.",
        "rake_selection_hold": "Temps de verrouillage par fixation du regard (plage : 0.0-5.0, défaut : 2.0)",
        "rake_selection_hold_desc": "Durée pendant laquelle le regard doit rester sur le même curseur avant qu’il se verrouille automatiquement. Utilisé seulement si le verrouillage est activé.",
        "rake_show_gaze": "Afficher le point de regard (Ninja uniquement, plage : off/on, défaut : on)",
        "rake_show_gaze_desc": "Affiche un marqueur rouge correspondant au regard estimé par la webcam.",
        "rake_without_targetfinder": "Sans TargetFinder (plage : off/on, défaut : on)",
        "rake_without_targetfinder_desc": "Lance Ninja Cursors(gaze) sans détection, sans surbrillance de cible et sans inférence du modèle. Seuls la sélection du curseur par le regard et les clics redirigés restent actifs.",
        "experiment_section": "Tâche expérimentale",
        "experiment_enabled": "Lancer la tâche expérimentale (plage : off/on, défaut : off)",
        "experiment_enabled_desc": "Si activé, Démarrer / Appliquer lance la tâche contrôlée de sélection de cibles sur captures d’écran au lieu d’une démo libre.",
        "experiment_data_dir": "Dossier du jeu de données (défaut : stage/data/web)",
        "experiment_data_dir_desc": "Dossier contenant les paires annotées .png/.txt utilisées pour générer les essais.",
        "experiment_trials": "Essais (plage : 1-1000, défaut : 12)",
        "experiment_trials_desc": "Nombre d’essais contrôlés de sélection de cible à exécuter.",
        "experiment_difficulty": "Difficulté (choix : easy/medium/hard/mixed, défaut : mixed)",
        "experiment_difficulty_desc": "Niveau échantillonné selon l’ID de Fitts : easy [0,3), medium [3,5), hard [5,8.5). Mixed échantillonne tous les niveaux.",
        "experiment_countdown": "Compte à rebours (secondes, plage : 0-30, défaut : 3)",
        "experiment_countdown_desc": "Secondes avant le début de chaque essai pendant que le curseur reste au centre de l’image.",
        "experiment_max_clicks": "Clics max par essai (plage : 1-20, défaut : 1)",
        "experiment_max_clicks_desc": "Nombre maximal de tentatives avant qu’un essai soit marqué comme échoué.",
        "experiment_fullscreen": "Expérience en plein écran (plage : off/on, défaut : on)",
        "experiment_fullscreen_desc": "Affiche la tâche sur capture d’écran en plein écran. À désactiver seulement pour le débogage.",
        "experiment_show_all_targets": "Afficher toutes les cibles annotées (debug, défaut : off)",
        "experiment_show_all_targets_desc": "Dessine toutes les boîtes d’annotation en vert. À utiliser seulement pour le débogage, pas pendant les passations.",
        "experiment_note": "Pour la condition TargetFinder, la tâche contrôlée utilise le jeu de données annoté comme vérité terrain et lance en interne le baseline souris.",
        "apply": "Démarrer / Appliquer",
        "change_thresh": "Seuil de changement (plage : 0-100000, défaut : 100)",
        "change_thresh_desc": "Plus haut = moins de rafraîchissements pour de petits changements. Plus bas = réaction plus rapide.",
        "model_path": "Modèle YOLO (fichier .pt) (défaut : yolo26s_1280.pt intégré)",
        "model_path_desc": "Laissez vide pour utiliser le fichier yolo26s_1280.pt intégré. Choisissez un autre fichier .pt entraîné pour changer de modèle.",
        "browse": "Parcourir",
        "use_default_model": "Utiliser le yolo26s_1280.pt intégré",
        "capture_interval": "Intervalle de capture (plage : 0.001-10.0, défaut : 0.033)",
        "capture_interval_desc": "Plus bas = vérifications plus rapides et plus de charge CPU/GPU. Plus haut = mises à jour plus lentes.",
        "confidence": "Confiance (plage : 0.0-1.0, défaut : 0.28)",
        "confidence_desc": "Plus bas = garde plus de détections. Plus haut = garde seulement les détections plus sûres.",
        "iou": "IoU (plage : 0.0-1.0, défaut : 0.3)",
        "iou_desc": "Plus bas = garde plus de boîtes qui se chevauchent. Plus haut = fusionne davantage les chevauchements.",
        "display": "Afficher le retour visuel (sémantique uniquement, plage : off/on, défaut : off)",
        "display_short": "Afficher le retour visuel",
        "display_desc": "Affiche les guides visuels du pointage sémantique quand c'est activé.",
        "disable_accel": "Désactiver l'accélération de la souris (sémantique uniquement, plage : off/on, défaut : off)",
        "disable_accel_short": "Désactiver l'accélération de la souris",
        "disable_accel_desc": "Rend le pointage sémantique plus stable, mais change la sensation de la souris pendant l'exécution.",
        "mode_note": "Overlay TargetFinder : affiche les boîtes détectées pour les tests. Bubble Cursor : agrandit la sélection autour de la cible la plus proche. Pointage sémantique : ralentit le pointeur près des cibles pour mieux viser. DynaSpot : garde le curseur système normal comme centre et agrandit une zone d’activation circulaire avec la vitesse tout en préservant les clics dans l’espace vide. Ninja Cursors(gaze) : le regard active d’abord le curseur le plus proche parmi 8 curseurs répartis ; si le regard y reste assez longtemps, ce curseur se verrouille automatiquement pour le micro-ajustement jusqu’au clic.",
        "contrast": "Contrast",
        "enable_tts": "Activer la synthèse vocale",
        "language": "Langue",
        "choose_language_dialog": "Choisir une langue",
        "confirm": "Confirmer",
        "cancel": "Annuler",
        "enter_value_for": "Saisissez une valeur pour {name}.",
        "turn_on": "Activer {name}",
        "turn_off": "Désactiver {name}",
        "stop": "Arrêter le mode en cours",
        "ready": "Prêt. Choisissez les réglages puis appuyez sur Démarrer / Appliquer.",
        "pending_apply": "Réglages mis à jour. Appuyez sur Démarrer / Appliquer pour lancer ou actualiser la technique choisie.",
        "select_mode_first": "Choisissez d'abord une technique puis appuyez sur Démarrer / Appliquer.",
        "running_bubble": "Bubble Cursor est en cours.",
        "running_semantic": "Le pointage sémantique est en cours.",
        "running_targetfinder": "L'overlay TargetFinder est en cours.",
        "running_dynaspot": "DynaSpot est en cours.",
        "running_rake": "Ninja Cursors(gaze) est en cours.",
        "running_experiment": "La tâche expérimentale est en cours.",
        "stopped": "Le mode en cours a été arrêté.",
        "no_running": "Aucun mode en cours n'a été trouvé.",
        "invalid_model_path": "Le fichier du modèle sélectionné est introuvable.",
        "invalid_experiment_data_dir": "Le dossier du jeu de données expérimental est introuvable.",
        "missing_one_euro": "OneEuroFilter n’est pas installé ou n’a pas pu être importé dans cet environnement.",
        "missing_webeyetrack": "WebEyeTrack n’est pas installé ou n’a pas pu être importé dans cet environnement.",
        "panel_updated": "L'apparence du panneau a été mise à jour.",
        "tts_enabled": "La synthèse vocale est activée.",
        "tts_disabled": "La synthèse vocale est désactivée.",
        "tts_unavailable": "La synthèse vocale n'est pas disponible sur ce système.",
        "language_updated": "La langue de l'interface a été mise à jour.",
        "q_hint": "Vous pouvez aussi appuyer sur q pour quitter le mode actif.",
    },
}


@dataclass
class PanelConfig:
    ignore_text: bool = False
    ignore_large_targets: bool = False
    show_bounding_boxes: bool = False
    show_class_labels: bool = False

    confidence: float = DEFAULT_CONFIDENCE
    change_thresh: int = DEFAULT_CHANGE_THRESH
    capture_interval: float = DEFAULT_CAPTURE_INTERVAL
    iou: float = DEFAULT_IOU
    model_path: str = ""
    filter_name: str = "none"
    filter_freq: float = DEFAULT_FILTER_FREQ
    filter_min_cutoff: float = DEFAULT_FILTER_MIN_CUTOFF
    filter_beta: float = DEFAULT_FILTER_BETA
    filter_d_cutoff: float = DEFAULT_FILTER_D_CUTOFF
    enable_logging: bool = False
    display: bool = False
    disable_accel: bool = False
    dynaspot_min_speed: float = DEFAULT_DYNASPOT_MIN_SPEED
    dynaspot_spot_width: float = DEFAULT_DYNASPOT_SPOT_WIDTH
    dynaspot_lag: float = DEFAULT_DYNASPOT_LAG
    dynaspot_reduce_time: float = DEFAULT_DYNASPOT_REDUCE_TIME
    rake_camera_index: int = DEFAULT_RAKE_CAMERA_INDEX
    rake_screen_width_cm: float = DEFAULT_RAKE_SCREEN_WIDTH_CM
    rake_screen_height_cm: float = DEFAULT_RAKE_SCREEN_HEIGHT_CM
    rake_spacing: float = DEFAULT_RAKE_SPACING
    rake_gaze_smoothing: float = DEFAULT_RAKE_GAZE_SMOOTHING
    rake_gaze_gain_x: float = DEFAULT_RAKE_GAZE_GAIN_X
    rake_gaze_gain_y: float = DEFAULT_RAKE_GAZE_GAIN_Y
    rake_gaze_offset_x: float = DEFAULT_RAKE_GAZE_OFFSET_X
    rake_gaze_offset_y: float = DEFAULT_RAKE_GAZE_OFFSET_Y
    rake_selection_hold: float = DEFAULT_RAKE_SELECTION_HOLD
    rake_lock_on_dwell: bool = DEFAULT_RAKE_LOCK_ON_DWELL
    rake_show_gaze: bool = True
    rake_without_targetfinder: bool = DEFAULT_RAKE_WITHOUT_TARGETFINDER
    rake_use_calibration: bool = DEFAULT_RAKE_USE_CALIBRATION
    rake_calib_points: int = DEFAULT_RAKE_CALIB_POINTS
    rake_auto_calibrate: bool = DEFAULT_RAKE_AUTO_CALIBRATE
    rake_calibration_status: str = "not_calibrated"
    experiment_enabled: bool = False
    experiment_data_dir: str = DEFAULT_EXPERIMENT_DATA_DIR
    experiment_trials: int = DEFAULT_EXPERIMENT_TRIALS
    experiment_difficulty: str = DEFAULT_EXPERIMENT_DIFFICULTY
    experiment_countdown: int = DEFAULT_EXPERIMENT_COUNTDOWN
    experiment_max_clicks: int = DEFAULT_EXPERIMENT_MAX_CLICKS
    experiment_fullscreen: bool = DEFAULT_EXPERIMENT_FULLSCREEN
    experiment_show_all_targets: bool = DEFAULT_EXPERIMENT_SHOW_ALL_TARGETS

    enable_bubble_cursor: bool = False
    enable_semantic_pointing: bool = False
    enable_dynaspot: bool = False
    enable_rake_cursor: bool = False

    high_contrast_mode: bool = False
    stronger_visual_cue: bool = False
    single_click_as_double_click: bool = False

    preset: str = ""
    enable_tts: bool = False
    language: str = "French"


class ControlPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TargetFinder Control Panel")
        self.resize(980, 700)
        self.setMinimumSize(860, 580)

        self.process = None
        self._speech_process = None
        self._mac_voice_names = None
        self._suspend_updates = True
        self._text_bindings = []
        self._focus_prompt_keys = {}
        self._help_prompt_keys = {}
        self._widget_speech_texts = {}
        self._last_auto_speech = ("", 0.0)
        self._hidden_config = PanelConfig()
        self._back_history = []
        self._forward_history = []
        self._prev_buttons = []
        self._next_buttons = []
        self._selected_mode = None
        self._selected_filter = "none"
        self._selected_language = "French"
        self._rake_calibration_status = "not_calibrated"
        self._rake_calibration_status_detail = None
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_path = self.project_root / "control_panel_config.json"
        self._process_watch_timer = QtCore.QTimer(self)
        self._process_watch_timer.setInterval(300)
        self._process_watch_timer.timeout.connect(self._poll_process_state)
        self._process_output_buffer = ""
        self._process_output_lines = []

        self._build_ui()
        self._connect_signals()
        self._apply_panel_style()
        self._load_if_exists()
        self._suspend_updates = False
        self._apply_language()
        self._update_mode_dependent_fields()
        self._update_history_buttons()
        self._set_status("ready")

    # -------------------------------
    # Translation helpers
    # -------------------------------
    def _language_code(self):
        return self._selected_language or "English"

    def _current_screen_physical_size_cm(self) -> tuple[float, float]:
        width_cm = None
        height_cm = None
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            physical_size = screen.physicalSize()
            width_cm = self._valid_screen_dimension_cm(physical_size.width() / 10.0)
            height_cm = self._valid_screen_dimension_cm(physical_size.height() / 10.0)
        return (
            width_cm if width_cm is not None else DEFAULT_RAKE_SCREEN_WIDTH_CM,
            height_cm if height_cm is not None else DEFAULT_RAKE_SCREEN_HEIGHT_CM,
        )

    @staticmethod
    def _valid_screen_dimension_cm(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if value <= 0.0:
            return None
        return value

    def _text(self, key: str) -> str:
        lang = self._language_code()
        return UI_TEXTS.get(lang, UI_TEXTS["English"]).get(key, key)

    def _format_text(self, key: str, **kwargs) -> str:
        return self._text(key).format(**kwargs)

    def _bind_text(self, widget, key: str):
        self._text_bindings.append((widget, key))
        widget.setText(self._text(key))

    def _apply_language(self):
        for widget, key in self._text_bindings:
            widget.setText(self._text(key))
        if hasattr(self, "mode_selector_button"):
            self._refresh_mode_selector_text()
        if hasattr(self, "filter_selector_button"):
            self._refresh_filter_selector_text()
        if hasattr(self, "language_selector_button"):
            self._refresh_language_selector_text()
        if hasattr(self, "model_path_edit"):
            self.model_path_edit.setPlaceholderText(self._text("use_default_model"))
        if hasattr(self, "rake_calibration_status_value"):
            self._update_rake_calibration_ui()

    def _set_status(self, key: str, *, speak: bool = False):
        message = self._text(key)
        self.info_label.setText(message)
        if speak:
            self._speak(message)

    def _speak_control_name(self, text: str):
        if text:
            self._speak(text)

    def _speak_auto_text(self, text: str, *, min_interval: float = 0.25):
        if not text:
            return
        last_text, last_at = self._last_auto_speech
        now = time.monotonic()
        if text == last_text and (now - last_at) < min_interval:
            return
        self._last_auto_speech = (text, now)
        self._speak(text)

    def _help_text(self, text_key: str, description_key: str | None = None, action_key: str | None = None):
        parts = [self._text(text_key)]
        if description_key:
            parts.append(self._text(description_key))
        if action_key:
            parts.append(self._text(action_key))
        return " ".join(part for part in parts if part)

    def _language_speech_label(self, code: str):
        return LANGUAGE_OPTIONS[code][self._language_code()]

    def _mode_label(self, code: str | None):
        if not code:
            return self._text("select_technique")
        return MODE_OPTIONS[code][self._language_code()]

    def _filter_label(self, code: str | None):
        if not code:
            return self._text("select_filter")
        key = f"filter_{code}"
        return self._text(key) if key in UI_TEXTS[self._language_code()] else code

    def _language_label(self, code: str):
        item = LANGUAGE_OPTIONS[code]
        return f"{item['badge']} {item[self._language_code()]}"

    def _refresh_mode_selector_text(self):
        self.mode_selector_button.setText(f"{self._mode_label(self._selected_mode)}  ▼")

    def _refresh_filter_selector_text(self):
        self.filter_selector_button.setText(f"{self._filter_label(self._selected_filter)}  ▼")

    def _refresh_language_selector_text(self):
        self.language_selector_button.setText(f"{self._language_label(self._selected_language)}  ▼")

    def _show_selection_dialog(self, title: str, options, current_value):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.resize(320, 240)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("DialogTitle")
        layout.addWidget(title_label)

        list_widget = QtWidgets.QListWidget()
        list_widget.setObjectName("SelectorList")
        for value, label in options:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, value)
            list_widget.addItem(item)
            if value == current_value:
                list_widget.setCurrentItem(item)
        layout.addWidget(list_widget, 1)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        cancel_button = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        if ok_button is not None:
            ok_button.setText(self._text("confirm"))
            ok_button.pressed.connect(lambda: self._speak_auto_text(self._text("confirm")))
        if cancel_button is not None:
            cancel_button.setText(self._text("cancel"))
            cancel_button.pressed.connect(lambda: self._speak_auto_text(self._text("cancel")))
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        def speak_current_item(item):
            if item is None:
                return
            self._speak_auto_text(item.text())

        list_widget.itemClicked.connect(speak_current_item)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None

        current_item = list_widget.currentItem()
        if current_item is None:
            return None
        return current_item.data(QtCore.Qt.ItemDataRole.UserRole)

    # -------------------------------
    # UI helpers
    # -------------------------------
    def _create_scroll_page(self):
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(18)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll, layout

    def _create_page_header(self, text_key: str):
        header = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        prev_button = QtWidgets.QToolButton()
        prev_button.setObjectName("HeaderNavButton")
        prev_button.setText("‹")
        prev_button.setFixedSize(42, 42)
        prev_button.clicked.connect(self._go_prev_page)
        self._prev_buttons.append(prev_button)

        next_button = QtWidgets.QToolButton()
        next_button.setObjectName("HeaderNavButton")
        next_button.setText("›")
        next_button.setFixedSize(42, 42)
        next_button.clicked.connect(self._go_next_page)
        self._next_buttons.append(next_button)

        title = QtWidgets.QLabel()
        title.setObjectName("PageTitle")
        self._bind_text(title, text_key)

        layout.addWidget(prev_button)
        layout.addWidget(next_button)
        layout.addWidget(title)
        layout.addStretch()
        return header

    def _create_card(self):
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(0)
        return card, layout

    def _create_setting_group(self, title_key: str, rows):
        group = QtWidgets.QFrame()
        group.setObjectName("SettingGroup")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(20, 14, 20, 16)
        layout.setSpacing(0)

        title = QtWidgets.QLabel()
        title.setObjectName("GroupTitle")
        title.setWordWrap(True)
        self._bind_text(title, title_key)
        layout.addWidget(title)

        for row in rows:
            layout.addWidget(row)
        return group

    def _create_switch(self):
        checkbox = QtWidgets.QCheckBox()
        checkbox.setText("")
        checkbox.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        return checkbox

    def _create_switch_row(self, text_key: str, widget: QtWidgets.QCheckBox, description_key: str | None = None):
        row = QtWidgets.QWidget()
        row.setObjectName("SettingRow")
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 14, 0, 14)
        layout.setSpacing(16)

        label_column = QtWidgets.QWidget()
        label_column.setObjectName("LabelColumn")
        label_layout = QtWidgets.QVBoxLayout(label_column)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(4)

        label = QtWidgets.QLabel()
        label.setObjectName("SettingLabel")
        label.setWordWrap(True)
        self._bind_text(label, text_key)
        label_layout.addWidget(label)

        if description_key is not None:
            description = QtWidgets.QLabel()
            description.setObjectName("SettingHelp")
            description.setWordWrap(True)
            self._bind_text(description, description_key)
            label_layout.addWidget(description)
        else:
            description = None

        layout.addWidget(label_column, 1)
        layout.addWidget(widget, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._register_help_targets(
            [row, label_column, label] + ([description] if description is not None else []),
            text_key,
            description_key,
        )
        return row

    def _create_field_row(self, text_key: str, widget, description_key: str | None = None):
        row = QtWidgets.QWidget()
        row.setObjectName("SettingRow")
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 14, 0, 14)
        layout.setSpacing(16)

        label_column = QtWidgets.QWidget()
        label_column.setObjectName("LabelColumn")
        label_layout = QtWidgets.QVBoxLayout(label_column)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(4)

        label = QtWidgets.QLabel()
        label.setObjectName("SettingLabel")
        label.setWordWrap(True)
        self._bind_text(label, text_key)
        label_layout.addWidget(label)

        if description_key is not None:
            description = QtWidgets.QLabel()
            description.setObjectName("SettingHelp")
            description.setWordWrap(True)
            self._bind_text(description, description_key)
            label_layout.addWidget(description)
        else:
            description = None

        if widget.objectName() == "SelectorButton":
            widget.setMinimumWidth(220)
            widget.setMaximumWidth(240)
        elif widget.objectName() == "ModelPicker":
            widget.setMinimumWidth(320)
            widget.setMaximumWidth(420)
        elif widget.objectName() == "CalibrationActions":
            widget.setMinimumWidth(320)
            widget.setMaximumWidth(420)
        else:
            widget.setMinimumWidth(150)
            widget.setMaximumWidth(170)

        layout.addWidget(label_column, 1)
        layout.addWidget(widget, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._register_help_targets(
            [row, label_column, label] + ([description] if description is not None else []),
            text_key,
            description_key,
        )
        return row

    def _create_label_value_row(self, text_key: str, value_widget, description_key: str | None = None):
        row = QtWidgets.QWidget()
        row.setObjectName("SettingRow")
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 14, 0, 14)
        layout.setSpacing(16)

        label_column = QtWidgets.QWidget()
        label_column.setObjectName("LabelColumn")
        label_layout = QtWidgets.QVBoxLayout(label_column)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(4)

        label = QtWidgets.QLabel()
        label.setObjectName("SettingLabel")
        label.setWordWrap(True)
        self._bind_text(label, text_key)
        label_layout.addWidget(label)

        if description_key is not None:
            description = QtWidgets.QLabel()
            description.setObjectName("SettingHelp")
            description.setWordWrap(True)
            self._bind_text(description, description_key)
            label_layout.addWidget(description)
        else:
            description = None

        value_widget.setMinimumWidth(220)
        layout.addWidget(label_column, 1)
        layout.addWidget(value_widget, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._register_help_targets(
            [row, label_column, label] + ([description] if description is not None else []),
            text_key,
            description_key,
        )
        return row

    def _create_note(self, text_key: str):
        label = QtWidgets.QLabel()
        label.setObjectName("SectionNote")
        label.setWordWrap(True)
        self._bind_text(label, text_key)
        return label

    def _create_separator(self):
        line = QtWidgets.QFrame()
        line.setObjectName("Separator")
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        return line

    # -----------------------------
    # Build main UI
    # -----------------------------
    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        self.sidebar_frame = QtWidgets.QFrame()
        self.sidebar_frame.setObjectName("SidebarFrame")
        self.sidebar_frame.setFixedWidth(250)

        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar_frame)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        nav_specs = [
            ("nav_mode", "top"),
            ("nav_accessibility", "middle"),
            ("nav_audio", "middle"),
            ("nav_language", "bottom"),
        ]
        self.nav_buttons = []
        for index, (text_key, role) in enumerate(nav_specs):
            button = QtWidgets.QPushButton()
            button.setCheckable(True)
            button.setObjectName("NavButton")
            button.setProperty("navRole", role)
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            button.clicked.connect(lambda checked, i=index: self._navigate_to_page(i))
            self._bind_text(button, text_key)
            self.nav_buttons.append(button)
            sidebar_layout.addWidget(button, 1)

        self.nav_buttons[0].setChecked(True)
        root.addWidget(self.sidebar_frame)

        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(14)
        root.addWidget(right_container, 1)

        self.pages = QtWidgets.QStackedWidget()
        right_layout.addWidget(self.pages, 1)

        self.pages.addWidget(self._build_mode_page())
        self.pages.addWidget(self._build_accessibility_page())
        self.pages.addWidget(self._build_audio_page())
        self.pages.addWidget(self._build_language_page())

        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setObjectName("InfoLabel")
        right_layout.addWidget(self.info_label)

        self.q_hint_label = QtWidgets.QLabel()
        self.q_hint_label.setWordWrap(True)
        self.q_hint_label.setObjectName("InfoLabel")
        self._bind_text(self.q_hint_label, "q_hint")
        right_layout.addWidget(self.q_hint_label)

        button_row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton()
        self.start_button.setObjectName("ActionButton")
        self._bind_text(self.start_button, "apply")
        self.stop_button = QtWidgets.QPushButton()
        self.stop_button.setObjectName("ActionButton")
        self._bind_text(self.stop_button, "stop")
        button_row.addStretch(1)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        right_layout.addLayout(button_row)

    # ------------------------
    # Pages
    # ------------------------
    def _build_mode_page(self):
        page, page_layout = self._create_scroll_page()
        page_layout.addWidget(self._create_page_header("page_mode"))

        self.mode_selector_button = QtWidgets.QPushButton()
        self.mode_selector_button.setObjectName("SelectorButton")
        self._refresh_mode_selector_text()

        self.filter_selector_button = QtWidgets.QPushButton()
        self.filter_selector_button.setObjectName("SelectorButton")
        self._refresh_filter_selector_text()

        self.filter_freq_spin = QtWidgets.QDoubleSpinBox()
        self.filter_freq_spin.setKeyboardTracking(False)
        self.filter_freq_spin.setDecimals(1)
        self.filter_freq_spin.setRange(1.0, 1000.0)
        self.filter_freq_spin.setSingleStep(10.0)
        self.filter_freq_spin.setValue(DEFAULT_FILTER_FREQ)

        self.filter_min_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.filter_min_cutoff_spin.setKeyboardTracking(False)
        self.filter_min_cutoff_spin.setDecimals(3)
        self.filter_min_cutoff_spin.setRange(0.001, 100.0)
        self.filter_min_cutoff_spin.setSingleStep(0.1)
        self.filter_min_cutoff_spin.setValue(DEFAULT_FILTER_MIN_CUTOFF)

        self.filter_beta_spin = QtWidgets.QDoubleSpinBox()
        self.filter_beta_spin.setKeyboardTracking(False)
        self.filter_beta_spin.setDecimals(3)
        self.filter_beta_spin.setRange(0.0, 10.0)
        self.filter_beta_spin.setSingleStep(0.01)
        self.filter_beta_spin.setValue(DEFAULT_FILTER_BETA)

        self.filter_d_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.filter_d_cutoff_spin.setKeyboardTracking(False)
        self.filter_d_cutoff_spin.setDecimals(3)
        self.filter_d_cutoff_spin.setRange(0.001, 100.0)
        self.filter_d_cutoff_spin.setSingleStep(0.1)
        self.filter_d_cutoff_spin.setValue(DEFAULT_FILTER_D_CUTOFF)

        self.model_path_edit = QtWidgets.QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText(self._text("use_default_model"))
        self.model_path_edit.setMinimumWidth(220)
        self.model_path_edit.setClearButtonEnabled(False)

        self.model_browse_button = QtWidgets.QPushButton()
        self.model_browse_button.setObjectName("SmallActionButton")
        self._bind_text(self.model_browse_button, "browse")

        self.model_picker = QtWidgets.QWidget()
        self.model_picker.setObjectName("ModelPicker")
        model_picker_layout = QtWidgets.QHBoxLayout(self.model_picker)
        model_picker_layout.setContentsMargins(0, 0, 0, 0)
        model_picker_layout.setSpacing(8)
        model_picker_layout.addWidget(self.model_path_edit, 1)
        model_picker_layout.addWidget(self.model_browse_button)

        self.change_thresh_spin = QtWidgets.QSpinBox()
        self.change_thresh_spin.setKeyboardTracking(False)
        self.change_thresh_spin.setRange(0, 100000)
        self.change_thresh_spin.setValue(DEFAULT_CHANGE_THRESH)

        self.capture_interval_spin = QtWidgets.QDoubleSpinBox()
        self.capture_interval_spin.setKeyboardTracking(False)
        self.capture_interval_spin.setDecimals(3)
        self.capture_interval_spin.setRange(0.001, 10.0)
        self.capture_interval_spin.setSingleStep(0.001)
        self.capture_interval_spin.setValue(DEFAULT_CAPTURE_INTERVAL)

        self.confidence_spin = QtWidgets.QDoubleSpinBox()
        self.confidence_spin.setKeyboardTracking(False)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setValue(DEFAULT_CONFIDENCE)

        self.iou_spin = QtWidgets.QDoubleSpinBox()
        self.iou_spin.setKeyboardTracking(False)
        self.iou_spin.setDecimals(1)
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setValue(DEFAULT_IOU)

        self.dynaspot_min_speed_spin = QtWidgets.QDoubleSpinBox()
        self.dynaspot_min_speed_spin.setKeyboardTracking(False)
        self.dynaspot_min_speed_spin.setDecimals(1)
        self.dynaspot_min_speed_spin.setRange(0.0, 5000.0)
        self.dynaspot_min_speed_spin.setSingleStep(10.0)
        self.dynaspot_min_speed_spin.setValue(DEFAULT_DYNASPOT_MIN_SPEED)

        self.dynaspot_spot_width_spin = QtWidgets.QDoubleSpinBox()
        self.dynaspot_spot_width_spin.setKeyboardTracking(False)
        self.dynaspot_spot_width_spin.setDecimals(1)
        self.dynaspot_spot_width_spin.setRange(1.0, 128.0)
        self.dynaspot_spot_width_spin.setSingleStep(1.0)
        self.dynaspot_spot_width_spin.setValue(DEFAULT_DYNASPOT_SPOT_WIDTH)

        self.dynaspot_lag_spin = QtWidgets.QDoubleSpinBox()
        self.dynaspot_lag_spin.setKeyboardTracking(False)
        self.dynaspot_lag_spin.setDecimals(3)
        self.dynaspot_lag_spin.setRange(0.0, 5.0)
        self.dynaspot_lag_spin.setSingleStep(0.01)
        self.dynaspot_lag_spin.setValue(DEFAULT_DYNASPOT_LAG)

        self.dynaspot_reduce_time_spin = QtWidgets.QDoubleSpinBox()
        self.dynaspot_reduce_time_spin.setKeyboardTracking(False)
        self.dynaspot_reduce_time_spin.setDecimals(3)
        self.dynaspot_reduce_time_spin.setRange(0.001, 10.0)
        self.dynaspot_reduce_time_spin.setSingleStep(0.01)
        self.dynaspot_reduce_time_spin.setValue(DEFAULT_DYNASPOT_REDUCE_TIME)

        self.rake_camera_index_spin = QtWidgets.QSpinBox()
        self.rake_camera_index_spin.setKeyboardTracking(False)
        self.rake_camera_index_spin.setRange(0, 10)
        self.rake_camera_index_spin.setValue(DEFAULT_RAKE_CAMERA_INDEX)

        detected_screen_width_cm, detected_screen_height_cm = self._current_screen_physical_size_cm()
        self.rake_screen_width_cm_spin = QtWidgets.QDoubleSpinBox()
        self.rake_screen_width_cm_spin.setKeyboardTracking(False)
        self.rake_screen_width_cm_spin.setDecimals(1)
        self.rake_screen_width_cm_spin.setRange(10.0, 200.0)
        self.rake_screen_width_cm_spin.setSingleStep(0.5)
        self.rake_screen_width_cm_spin.setValue(detected_screen_width_cm)

        self.rake_screen_height_cm_spin = QtWidgets.QDoubleSpinBox()
        self.rake_screen_height_cm_spin.setKeyboardTracking(False)
        self.rake_screen_height_cm_spin.setDecimals(1)
        self.rake_screen_height_cm_spin.setRange(10.0, 200.0)
        self.rake_screen_height_cm_spin.setSingleStep(0.5)
        self.rake_screen_height_cm_spin.setValue(detected_screen_height_cm)

        self.rake_spacing_spin = QtWidgets.QDoubleSpinBox()
        self.rake_spacing_spin.setKeyboardTracking(False)
        self.rake_spacing_spin.setDecimals(1)
        self.rake_spacing_spin.setRange(80.0, 800.0)
        self.rake_spacing_spin.setSingleStep(10.0)
        self.rake_spacing_spin.setValue(DEFAULT_RAKE_SPACING)

        self.rake_gaze_smoothing_spin = QtWidgets.QDoubleSpinBox()
        self.rake_gaze_smoothing_spin.setKeyboardTracking(False)
        self.rake_gaze_smoothing_spin.setDecimals(2)
        self.rake_gaze_smoothing_spin.setRange(0.0, 0.95)
        self.rake_gaze_smoothing_spin.setSingleStep(0.01)
        self.rake_gaze_smoothing_spin.setValue(DEFAULT_RAKE_GAZE_SMOOTHING)

        self.rake_gaze_gain_x_spin = QtWidgets.QDoubleSpinBox()
        self.rake_gaze_gain_x_spin.setKeyboardTracking(False)
        self.rake_gaze_gain_x_spin.setDecimals(2)
        self.rake_gaze_gain_x_spin.setRange(0.1, 10.0)
        self.rake_gaze_gain_x_spin.setSingleStep(0.05)
        self.rake_gaze_gain_x_spin.setValue(DEFAULT_RAKE_GAZE_GAIN_X)

        self.rake_gaze_gain_y_spin = QtWidgets.QDoubleSpinBox()
        self.rake_gaze_gain_y_spin.setKeyboardTracking(False)
        self.rake_gaze_gain_y_spin.setDecimals(2)
        self.rake_gaze_gain_y_spin.setRange(0.1, 10.0)
        self.rake_gaze_gain_y_spin.setSingleStep(0.05)
        self.rake_gaze_gain_y_spin.setValue(DEFAULT_RAKE_GAZE_GAIN_Y)

        self.rake_gaze_offset_x_spin = QtWidgets.QDoubleSpinBox()
        self.rake_gaze_offset_x_spin.setKeyboardTracking(False)
        self.rake_gaze_offset_x_spin.setDecimals(1)
        self.rake_gaze_offset_x_spin.setRange(-1000.0, 1000.0)
        self.rake_gaze_offset_x_spin.setSingleStep(5.0)
        self.rake_gaze_offset_x_spin.setValue(DEFAULT_RAKE_GAZE_OFFSET_X)

        self.rake_gaze_offset_y_spin = QtWidgets.QDoubleSpinBox()
        self.rake_gaze_offset_y_spin.setKeyboardTracking(False)
        self.rake_gaze_offset_y_spin.setDecimals(1)
        self.rake_gaze_offset_y_spin.setRange(-1000.0, 1000.0)
        self.rake_gaze_offset_y_spin.setSingleStep(5.0)
        self.rake_gaze_offset_y_spin.setValue(DEFAULT_RAKE_GAZE_OFFSET_Y)

        self.rake_selection_hold_spin = QtWidgets.QDoubleSpinBox()
        self.rake_selection_hold_spin.setKeyboardTracking(False)
        self.rake_selection_hold_spin.setDecimals(2)
        self.rake_selection_hold_spin.setRange(0.0, 5.0)
        self.rake_selection_hold_spin.setSingleStep(0.05)
        self.rake_selection_hold_spin.setValue(DEFAULT_RAKE_SELECTION_HOLD)

        self.display_cb = self._create_switch()
        self.disable_accel_cb = self._create_switch()
        self.log_data_cb = self._create_switch()
        self.rake_lock_on_dwell_cb = self._create_switch()
        self.rake_show_gaze_cb = self._create_switch()
        self.rake_without_targetfinder_cb = self._create_switch()
        self.rake_use_calibration_cb = self._create_switch()
        self.rake_calib_points_combo = QtWidgets.QComboBox()
        self.rake_calib_points_combo.addItems(["5", "9", "13"])
        self.rake_calib_points_combo.setCurrentText(str(DEFAULT_RAKE_CALIB_POINTS))

        self.rake_reset_calibration_button = QtWidgets.QPushButton()
        self.rake_reset_calibration_button.setObjectName("SmallActionButton")
        self._bind_text(self.rake_reset_calibration_button, "rake_reset_calibration")

        self.rake_calibration_actions_widget = QtWidgets.QWidget()
        self.rake_calibration_actions_widget.setObjectName("CalibrationActions")
        actions_layout = QtWidgets.QHBoxLayout(self.rake_calibration_actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(8)
        actions_layout.addWidget(self.rake_reset_calibration_button)

        self.rake_calibration_status_value = QtWidgets.QLabel()
        self.rake_calibration_status_value.setObjectName("SettingValueLabel")
        self.rake_calibration_status_value.setWordWrap(True)

        self.rake_calibration_mode_value = QtWidgets.QLabel()
        self.rake_calibration_mode_value.setObjectName("SettingValueLabel")
        self.rake_calibration_mode_value.setWordWrap(True)

        self.rake_calibration_note = QtWidgets.QLabel()
        self.rake_calibration_note.setObjectName("SectionNote")
        self.rake_calibration_note.setWordWrap(True)

        self.experiment_enabled_cb = self._create_switch()

        self.experiment_data_path_edit = QtWidgets.QLineEdit()
        self.experiment_data_path_edit.setReadOnly(True)
        self.experiment_data_path_edit.setText(DEFAULT_EXPERIMENT_DATA_DIR)
        self.experiment_data_path_edit.setMinimumWidth(220)
        self.experiment_data_path_edit.setClearButtonEnabled(False)

        self.experiment_browse_button = QtWidgets.QPushButton()
        self.experiment_browse_button.setObjectName("SmallActionButton")
        self._bind_text(self.experiment_browse_button, "browse")

        self.experiment_data_picker = QtWidgets.QWidget()
        self.experiment_data_picker.setObjectName("ModelPicker")
        experiment_data_layout = QtWidgets.QHBoxLayout(self.experiment_data_picker)
        experiment_data_layout.setContentsMargins(0, 0, 0, 0)
        experiment_data_layout.setSpacing(8)
        experiment_data_layout.addWidget(self.experiment_data_path_edit, 1)
        experiment_data_layout.addWidget(self.experiment_browse_button)

        self.experiment_trials_spin = QtWidgets.QSpinBox()
        self.experiment_trials_spin.setKeyboardTracking(False)
        self.experiment_trials_spin.setRange(1, 1000)
        self.experiment_trials_spin.setValue(DEFAULT_EXPERIMENT_TRIALS)

        self.experiment_difficulty_combo = QtWidgets.QComboBox()
        self.experiment_difficulty_combo.addItems(["mixed", "easy", "medium", "hard"])
        self.experiment_difficulty_combo.setCurrentText(DEFAULT_EXPERIMENT_DIFFICULTY)

        self.experiment_countdown_spin = QtWidgets.QSpinBox()
        self.experiment_countdown_spin.setKeyboardTracking(False)
        self.experiment_countdown_spin.setRange(0, 30)
        self.experiment_countdown_spin.setValue(DEFAULT_EXPERIMENT_COUNTDOWN)

        self.experiment_max_clicks_spin = QtWidgets.QSpinBox()
        self.experiment_max_clicks_spin.setKeyboardTracking(False)
        self.experiment_max_clicks_spin.setRange(1, 20)
        self.experiment_max_clicks_spin.setValue(DEFAULT_EXPERIMENT_MAX_CLICKS)

        self.experiment_fullscreen_cb = self._create_switch()
        self.experiment_fullscreen_cb.setChecked(DEFAULT_EXPERIMENT_FULLSCREEN)
        self.experiment_show_all_targets_cb = self._create_switch()

        self._semantic_rows = [
            self._create_separator(),
            self._create_switch_row("display", self.display_cb, "display_desc"),
            self._create_separator(),
            self._create_switch_row("disable_accel", self.disable_accel_cb, "disable_accel_desc"),
        ]

        self._dynaspot_rows = [
            self._create_separator(),
            self._create_field_row("dynaspot_min_speed", self.dynaspot_min_speed_spin, "dynaspot_min_speed_desc"),
            self._create_separator(),
            self._create_field_row("dynaspot_spot_width", self.dynaspot_spot_width_spin, "dynaspot_spot_width_desc"),
            self._create_separator(),
            self._create_field_row("dynaspot_lag", self.dynaspot_lag_spin, "dynaspot_lag_desc"),
            self._create_separator(),
            self._create_field_row("dynaspot_reduce_time", self.dynaspot_reduce_time_spin, "dynaspot_reduce_time_desc"),
        ]

        self._filter_param_rows = [
            self._create_separator(),
            self._create_field_row("filter_freq", self.filter_freq_spin, "filter_freq_desc"),
            self._create_separator(),
            self._create_field_row("filter_min_cutoff", self.filter_min_cutoff_spin, "filter_min_cutoff_desc"),
            self._create_separator(),
            self._create_field_row("filter_beta", self.filter_beta_spin, "filter_beta_desc"),
            self._create_separator(),
            self._create_field_row("filter_d_cutoff", self.filter_d_cutoff_spin, "filter_d_cutoff_desc"),
        ]

        self._rake_rows = [
            self._create_separator(),
            self._create_field_row("rake_camera_index", self.rake_camera_index_spin, "rake_camera_index_desc"),
            self._create_separator(),
            self._create_field_row("rake_screen_width_cm", self.rake_screen_width_cm_spin, "rake_screen_width_cm_desc"),
            self._create_separator(),
            self._create_field_row("rake_screen_height_cm", self.rake_screen_height_cm_spin, "rake_screen_height_cm_desc"),
            self._create_separator(),
            self._create_field_row("rake_spacing", self.rake_spacing_spin, "rake_spacing_desc"),
            self._create_separator(),
            self._create_field_row("rake_gaze_smoothing", self.rake_gaze_smoothing_spin, "rake_gaze_smoothing_desc"),
            self._create_separator(),
            self._create_switch_row("rake_use_calibration", self.rake_use_calibration_cb, "rake_use_calibration_desc"),
            self._create_separator(),
            self._create_field_row("rake_calibration_points", self.rake_calib_points_combo, "rake_calibration_points_desc"),
            self._create_separator(),
            self._create_field_row("rake_calibration_actions", self.rake_calibration_actions_widget, "rake_calibration_actions_desc"),
            self._create_separator(),
            self._create_label_value_row("rake_calibration_status", self.rake_calibration_status_value, "rake_calibration_status_desc"),
            self._create_separator(),
            self._create_label_value_row("rake_calibration_mode", self.rake_calibration_mode_value, "rake_calibration_mode_desc"),
            self._create_separator(),
            self.rake_calibration_note,
            self._create_separator(),
            self._create_field_row("rake_gaze_gain_x", self.rake_gaze_gain_x_spin, "rake_gaze_gain_x_desc"),
            self._create_separator(),
            self._create_field_row("rake_gaze_gain_y", self.rake_gaze_gain_y_spin, "rake_gaze_gain_y_desc"),
            self._create_separator(),
            self._create_field_row("rake_gaze_offset_x", self.rake_gaze_offset_x_spin, "rake_gaze_offset_x_desc"),
            self._create_separator(),
            self._create_field_row("rake_gaze_offset_y", self.rake_gaze_offset_y_spin, "rake_gaze_offset_y_desc"),
            self._create_separator(),
            self._create_switch_row("rake_lock_on_dwell", self.rake_lock_on_dwell_cb, "rake_lock_on_dwell_desc"),
            self._create_separator(),
            self._create_field_row("rake_selection_hold", self.rake_selection_hold_spin, "rake_selection_hold_desc"),
            self._create_separator(),
            self._create_switch_row("rake_show_gaze", self.rake_show_gaze_cb, "rake_show_gaze_desc"),
            self._create_separator(),
            self._create_switch_row("rake_without_targetfinder", self.rake_without_targetfinder_cb, "rake_without_targetfinder_desc"),
        ]

        self._experiment_param_rows = [
            self._create_separator(),
            self._create_field_row("experiment_data_dir", self.experiment_data_picker, "experiment_data_dir_desc"),
            self._create_separator(),
            self._create_field_row("experiment_trials", self.experiment_trials_spin, "experiment_trials_desc"),
            self._create_separator(),
            self._create_field_row("experiment_difficulty", self.experiment_difficulty_combo, "experiment_difficulty_desc"),
            self._create_separator(),
            self._create_field_row("experiment_countdown", self.experiment_countdown_spin, "experiment_countdown_desc"),
            self._create_separator(),
            self._create_field_row("experiment_max_clicks", self.experiment_max_clicks_spin, "experiment_max_clicks_desc"),
            self._create_separator(),
            self._create_switch_row("experiment_fullscreen", self.experiment_fullscreen_cb, "experiment_fullscreen_desc"),
            self._create_separator(),
            self._create_switch_row("experiment_show_all_targets", self.experiment_show_all_targets_cb, "experiment_show_all_targets_desc"),
            self._create_separator(),
            self._create_note("experiment_note"),
        ]

        setup_rows = [
            self._create_field_row("model_path", self.model_picker, "model_path_desc"),
            self._create_separator(),
            self._create_field_row("technique", self.mode_selector_button, "mode_note"),
        ]
        experiment_toggle_rows = [
            self._create_switch_row("experiment_enabled", self.experiment_enabled_cb, "experiment_enabled_desc"),
            *self._experiment_param_rows,
        ]
        filter_rows = [
            self._create_field_row("filter", self.filter_selector_button, "filter_desc"),
            *self._filter_param_rows,
        ]
        logging_rows = [
            self._create_switch_row("record_data", self.log_data_cb, "record_data_desc"),
        ]
        detection_rows = [
            self._create_field_row("change_thresh", self.change_thresh_spin, "change_thresh_desc"),
            self._create_separator(),
            self._create_field_row("capture_interval", self.capture_interval_spin, "capture_interval_desc"),
            self._create_separator(),
            self._create_field_row("confidence", self.confidence_spin, "confidence_desc"),
            self._create_separator(),
            self._create_field_row("iou", self.iou_spin, "iou_desc"),
        ]

        self._experiment_param_group = None
        self._semantic_group = self._create_setting_group("semantic_section", self._semantic_rows)
        self._dynaspot_group = self._create_setting_group("dynaspot_section", self._dynaspot_rows)
        self._rake_group = self._create_setting_group("rake_section", self._rake_rows)

        groups = [
            self._create_setting_group("setup_section", setup_rows),
            self._create_setting_group("filter_section", filter_rows),
            self._create_setting_group("logging_section", logging_rows),
            self._create_setting_group("detection_section", detection_rows),
            self._create_setting_group("experiment_section", experiment_toggle_rows),
            self._semantic_group,
            self._dynaspot_group,
            self._rake_group,
        ]

        for group in groups:
            page_layout.addWidget(group)
        page_layout.addStretch()
        return page

    def _build_accessibility_page(self):
        page, page_layout = self._create_scroll_page()
        page_layout.addWidget(self._create_page_header("page_accessibility"))

        card, card_layout = self._create_card()
        self.high_contrast_cb = self._create_switch()
        card_layout.addWidget(self._create_switch_row("contrast", self.high_contrast_cb))

        page_layout.addWidget(card)
        page_layout.addStretch()
        return page

    def _build_audio_page(self):
        page, page_layout = self._create_scroll_page()
        page_layout.addWidget(self._create_page_header("page_audio"))

        card, card_layout = self._create_card()
        self.enable_tts_cb = self._create_switch()
        card_layout.addWidget(self._create_switch_row("enable_tts", self.enable_tts_cb))

        page_layout.addWidget(card)
        page_layout.addStretch()
        return page

    def _build_language_page(self):
        page, page_layout = self._create_scroll_page()
        page_layout.addWidget(self._create_page_header("page_language"))

        card, card_layout = self._create_card()
        self.language_selector_button = QtWidgets.QPushButton()
        self.language_selector_button.setObjectName("SelectorButton")
        self._refresh_language_selector_text()
        card_layout.addWidget(self._create_field_row("language", self.language_selector_button))

        page_layout.addWidget(card)
        page_layout.addStretch()
        return page

    # --------------------------------
    # Connections
    # --------------------------------
    def _connect_signals(self):
        self.mode_selector_button.clicked.connect(self._handle_mode_selection)
        self.filter_selector_button.clicked.connect(self._handle_filter_selection)
        self.language_selector_button.clicked.connect(self._handle_language_selection)
        self.model_browse_button.clicked.connect(self._handle_model_browse)
        self.experiment_browse_button.clicked.connect(self._handle_experiment_data_browse)
        self.start_button.clicked.connect(self._handle_apply_clicked)
        self.stop_button.clicked.connect(self._stop_demo)
        self.change_thresh_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.capture_interval_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.confidence_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.iou_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.filter_freq_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.filter_min_cutoff_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.filter_beta_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.filter_d_cutoff_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.dynaspot_min_speed_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.dynaspot_spot_width_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.dynaspot_lag_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.dynaspot_reduce_time_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.rake_camera_index_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.rake_screen_width_cm_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.rake_screen_height_cm_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.rake_spacing_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.rake_gaze_smoothing_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.rake_gaze_gain_x_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.rake_gaze_gain_y_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.rake_gaze_offset_x_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.rake_gaze_offset_y_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.rake_selection_hold_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.display_cb.toggled.connect(self._handle_runtime_option_change)
        self.disable_accel_cb.toggled.connect(self._handle_runtime_option_change)
        self.log_data_cb.toggled.connect(self._handle_runtime_option_change)
        self.rake_lock_on_dwell_cb.toggled.connect(self._handle_runtime_option_change)
        self.rake_show_gaze_cb.toggled.connect(self._handle_runtime_option_change)
        self.rake_without_targetfinder_cb.toggled.connect(self._handle_runtime_option_change)
        self.rake_use_calibration_cb.toggled.connect(self._handle_rake_calibration_toggle)
        self.rake_calib_points_combo.currentIndexChanged.connect(self._handle_runtime_option_change)
        self.rake_reset_calibration_button.clicked.connect(self._handle_rake_reset_calibration)
        self.experiment_enabled_cb.toggled.connect(self._handle_runtime_option_change)
        self.experiment_trials_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.experiment_difficulty_combo.currentIndexChanged.connect(self._handle_runtime_option_change)
        self.experiment_countdown_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.experiment_max_clicks_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.experiment_fullscreen_cb.toggled.connect(self._handle_runtime_option_change)
        self.experiment_show_all_targets_cb.toggled.connect(self._handle_runtime_option_change)

        self.high_contrast_cb.toggled.connect(self._handle_panel_style_change)
        self.enable_tts_cb.toggled.connect(self._handle_tts_change)

        self._register_numeric_field(self.change_thresh_spin, "change_thresh")
        self._register_numeric_field(self.capture_interval_spin, "capture_interval")
        self._register_numeric_field(self.confidence_spin, "confidence")
        self._register_numeric_field(self.iou_spin, "iou")
        self._register_numeric_field(self.filter_freq_spin, "filter_freq")
        self._register_numeric_field(self.filter_min_cutoff_spin, "filter_min_cutoff")
        self._register_numeric_field(self.filter_beta_spin, "filter_beta")
        self._register_numeric_field(self.filter_d_cutoff_spin, "filter_d_cutoff")
        self._register_numeric_field(self.dynaspot_min_speed_spin, "dynaspot_min_speed")
        self._register_numeric_field(self.dynaspot_spot_width_spin, "dynaspot_spot_width")
        self._register_numeric_field(self.dynaspot_lag_spin, "dynaspot_lag")
        self._register_numeric_field(self.dynaspot_reduce_time_spin, "dynaspot_reduce_time")
        self._register_numeric_field(self.rake_camera_index_spin, "rake_camera_index")
        self._register_numeric_field(self.rake_screen_width_cm_spin, "rake_screen_width_cm")
        self._register_numeric_field(self.rake_screen_height_cm_spin, "rake_screen_height_cm")
        self._register_numeric_field(self.rake_spacing_spin, "rake_spacing")
        self._register_numeric_field(self.rake_gaze_smoothing_spin, "rake_gaze_smoothing")
        self._register_numeric_field(self.rake_gaze_gain_x_spin, "rake_gaze_gain_x")
        self._register_numeric_field(self.rake_gaze_gain_y_spin, "rake_gaze_gain_y")
        self._register_numeric_field(self.rake_gaze_offset_x_spin, "rake_gaze_offset_x")
        self._register_numeric_field(self.rake_gaze_offset_y_spin, "rake_gaze_offset_y")
        self._register_numeric_field(self.rake_selection_hold_spin, "rake_selection_hold")
        self._register_numeric_field(self.experiment_trials_spin, "experiment_trials")
        self._register_numeric_field(self.experiment_countdown_spin, "experiment_countdown")
        self._register_numeric_field(self.experiment_max_clicks_spin, "experiment_max_clicks")
        self._register_help_targets(
            [self.model_picker, self.model_path_edit, self.model_browse_button],
            "model_path",
            "model_path_desc",
        )
        self._register_help_targets(
            [self.experiment_data_picker, self.experiment_data_path_edit, self.experiment_browse_button],
            "experiment_data_dir",
            "experiment_data_dir_desc",
        )
        self._register_help_targets([self.filter_selector_button], "filter", "filter_desc")
        self._register_help_targets([self.filter_freq_spin], "filter_freq", "filter_freq_desc")
        self._register_help_targets([self.filter_min_cutoff_spin], "filter_min_cutoff", "filter_min_cutoff_desc")
        self._register_help_targets([self.filter_beta_spin], "filter_beta", "filter_beta_desc")
        self._register_help_targets([self.filter_d_cutoff_spin], "filter_d_cutoff", "filter_d_cutoff_desc")

    # ---------------------------------
    # Navigation
    # ---------------------------------
    def _set_page(self, index: int):
        self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)
        is_mode_page = index == 0
        self.start_button.setVisible(is_mode_page)
        self.stop_button.setVisible(is_mode_page)
        self.q_hint_label.setVisible(is_mode_page and self._mode_code() is not None)

    def _navigate_to_page(self, index: int):
        current = self.pages.currentIndex()
        if index == current:
            return
        self._back_history.append(current)
        self._forward_history.clear()
        self._set_page(index)
        self._update_history_buttons()
        self._speak_control_name(self.nav_buttons[index].text())

    def _update_history_buttons(self):
        for button in self._prev_buttons:
            button.setEnabled(bool(self._back_history))
        for button in self._next_buttons:
            button.setEnabled(bool(self._forward_history))

    def _go_prev_page(self):
        if not self._back_history:
            return
        current = self.pages.currentIndex()
        index = self._back_history.pop()
        self._forward_history.append(current)
        self._set_page(index)
        self._update_history_buttons()
        self._speak_control_name(self.nav_buttons[index].text())

    def _go_next_page(self):
        if not self._forward_history:
            return
        current = self.pages.currentIndex()
        index = self._forward_history.pop()
        self._back_history.append(current)
        self._set_page(index)
        self._update_history_buttons()
        self._speak_control_name(self.nav_buttons[index].text())

    # -----------------------------------
    # Logic
    # -----------------------------------
    def _mode_code(self):
        return self._selected_mode

    def _update_action_buttons(self):
        self.start_button.setEnabled(self._mode_code() is not None and not self._is_demo_running())

    def _update_mode_dependent_fields(self):
        semantic_enabled = self._mode_code() == "semantic"
        dynaspot_enabled = self._mode_code() == "dynaspot"
        rake_enabled = self._mode_code() == "rake"
        filter_params_visible = self._selected_filter == "one_euro"
        experiment_enabled = self.experiment_enabled_cb.isChecked()
        self.display_cb.setEnabled(semantic_enabled)
        self.disable_accel_cb.setEnabled(semantic_enabled)
        if getattr(self, "_experiment_param_group", None) is not None:
            self._experiment_param_group.setVisible(experiment_enabled)
        if hasattr(self, "_semantic_group"):
            self._semantic_group.setVisible(semantic_enabled)
        if hasattr(self, "_dynaspot_group"):
            self._dynaspot_group.setVisible(dynaspot_enabled)
        if hasattr(self, "_rake_group"):
            self._rake_group.setVisible(rake_enabled)
        for row in getattr(self, "_experiment_param_rows", []):
            row.setVisible(experiment_enabled)
        for widget in (
            self.experiment_data_picker,
            self.experiment_data_path_edit,
            self.experiment_browse_button,
            self.experiment_trials_spin,
            self.experiment_difficulty_combo,
            self.experiment_countdown_spin,
            self.experiment_max_clicks_spin,
            self.experiment_fullscreen_cb,
            self.experiment_show_all_targets_cb,
        ):
            widget.setEnabled(experiment_enabled)
        for row in getattr(self, "_filter_param_rows", []):
            row.setVisible(filter_params_visible)
        for widget in (
            self.filter_freq_spin,
            self.filter_min_cutoff_spin,
            self.filter_beta_spin,
            self.filter_d_cutoff_spin,
        ):
            widget.setEnabled(filter_params_visible)
        for row in getattr(self, "_semantic_rows", []):
            row.setVisible(semantic_enabled)
        for row in getattr(self, "_dynaspot_rows", []):
            row.setVisible(dynaspot_enabled)
        for row in getattr(self, "_rake_rows", []):
            row.setVisible(rake_enabled)
        self._update_rake_calibration_ui(rake_enabled=rake_enabled)
        self._update_action_buttons()
        self.q_hint_label.setVisible(self.pages.currentIndex() == 0 and self._mode_code() is not None)

    def _calibration_status_text_key(self) -> str:
        return {
            "calibrating": "rake_calibration_status_calibrating",
            "calibrated": "rake_calibration_status_calibrated",
            "failed": "rake_calibration_status_failed",
            "cancelled": "rake_calibration_status_cancelled",
            "last_applied": "rake_calibration_status_last_applied",
        }.get(self._rake_calibration_status, "rake_calibration_status_not_calibrated")

    def _update_rake_calibration_ui(self, *, rake_enabled: bool | None = None):
        if rake_enabled is None:
            rake_enabled = self._mode_code() == "rake"
        use_calibration = self.rake_use_calibration_cb.isChecked()
        for widget in (
            self.rake_gaze_gain_x_spin,
            self.rake_gaze_gain_y_spin,
            self.rake_gaze_offset_x_spin,
            self.rake_gaze_offset_y_spin,
        ):
            widget.setEnabled(rake_enabled)
        self.rake_lock_on_dwell_cb.setEnabled(rake_enabled)
        self.rake_selection_hold_spin.setEnabled(rake_enabled and self.rake_lock_on_dwell_cb.isChecked())
        self.rake_calib_points_combo.setEnabled(rake_enabled and use_calibration)
        self.rake_reset_calibration_button.setEnabled(rake_enabled)
        status_text = self._text(self._calibration_status_text_key())
        if self._rake_calibration_status_detail:
            status_text = f"{status_text} ({self._rake_calibration_status_detail})"
        self.rake_calibration_status_value.setText(status_text)
        self.rake_calibration_mode_value.setText(
            self._text("rake_calibration_mode_active" if use_calibration else "rake_calibration_mode_manual")
        )
        self.rake_calibration_note.setText(
            self._text("rake_calibration_manual_disabled" if use_calibration else "rake_calibration_manual_enabled")
        )

    def _register_numeric_field(self, widget, text_key: str):
        self._focus_prompt_keys[widget] = text_key
        widget.installEventFilter(self)
        line_edit = widget.lineEdit()
        if line_edit is not None:
            self._focus_prompt_keys[line_edit] = text_key
            line_edit.installEventFilter(self)
        widget.editingFinished.connect(lambda w=widget: self._handle_numeric_edit_finished(w))

    def _register_help_targets(self, widgets, text_key: str, description_key: str | None = None):
        for widget in widgets:
            if widget is None:
                continue
            self._help_prompt_keys[widget] = (text_key, description_key)
            widget.installEventFilter(self)

    def _register_widget_speech(self, widget, text: str):
        if widget is None:
            return
        self._widget_speech_texts[widget] = text
        widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.FocusIn:
            text_key = self._focus_prompt_keys.get(watched)
            if text_key:
                self._speak_auto_text(
                    self._format_text("enter_value_for", name=self._text(text_key))
                )
            direct_text = self._widget_speech_texts.get(watched)
            if direct_text:
                self._speak_auto_text(direct_text)
        elif event.type() == QtCore.QEvent.Type.MouseButtonPress:
            help_keys = self._help_prompt_keys.get(watched)
            if help_keys:
                text_key, description_key = help_keys
                self._speak_auto_text(self._help_text(text_key, description_key))
            direct_text = self._widget_speech_texts.get(watched)
            if direct_text:
                self._speak_auto_text(direct_text)
        return super().eventFilter(watched, event)

    def _handle_mode_selection(self):
        self._speak_auto_text(self._text("select_technique"))
        options = [
            ("targetfinder", self._mode_label("targetfinder")),
            ("bubble", self._mode_label("bubble")),
            ("semantic", self._mode_label("semantic")),
            ("dynaspot", self._mode_label("dynaspot")),
            ("rake", self._mode_label("rake")),
        ]
        selected = self._show_selection_dialog(
            self._text("choose_mode_dialog"),
            options,
            self._selected_mode,
        )
        if selected is None:
            return
        self._selected_mode = selected
        self._refresh_mode_selector_text()
        self._update_mode_dependent_fields()
        self._save_config()
        self._set_status("pending_apply")

    def _handle_filter_selection(self):
        self._speak_auto_text(self._text("select_filter"))
        options = [(key, self._filter_label(key)) for key in FILTER_OPTIONS]
        selected = self._show_selection_dialog(
            self._text("choose_filter_dialog"),
            options,
            self._selected_filter,
        )
        if selected is None:
            return
        self._selected_filter = selected
        self._refresh_filter_selector_text()
        self._update_mode_dependent_fields()
        self._save_config()
        self._set_status("pending_apply")

    def _handle_runtime_option_change(self, *_args):
        if self._suspend_updates:
            return
        self._save_config()
        if self.sender() is self.experiment_enabled_cb:
            self._update_mode_dependent_fields()
        self._update_rake_calibration_ui()
        self._set_status("pending_apply")
        sender = self.sender()
        if sender is self.change_thresh_spin:
            self._speak_auto_text(self.change_thresh_spin.text())
        elif sender is self.capture_interval_spin:
            self._speak_auto_text(self.capture_interval_spin.text())
        elif sender is self.confidence_spin:
            self._speak_auto_text(self.confidence_spin.text())
        elif sender is self.iou_spin:
            self._speak_auto_text(self.iou_spin.text())
        elif sender is self.filter_freq_spin:
            self._speak_auto_text(self.filter_freq_spin.text())
        elif sender is self.filter_min_cutoff_spin:
            self._speak_auto_text(self.filter_min_cutoff_spin.text())
        elif sender is self.filter_beta_spin:
            self._speak_auto_text(self.filter_beta_spin.text())
        elif sender is self.filter_d_cutoff_spin:
            self._speak_auto_text(self.filter_d_cutoff_spin.text())
        elif sender is self.dynaspot_min_speed_spin:
            self._speak_auto_text(self.dynaspot_min_speed_spin.text())
        elif sender is self.dynaspot_spot_width_spin:
            self._speak_auto_text(self.dynaspot_spot_width_spin.text())
        elif sender is self.dynaspot_lag_spin:
            self._speak_auto_text(self.dynaspot_lag_spin.text())
        elif sender is self.dynaspot_reduce_time_spin:
            self._speak_auto_text(self.dynaspot_reduce_time_spin.text())
        elif sender is self.rake_camera_index_spin:
            self._speak_auto_text(self.rake_camera_index_spin.text())
        elif sender is self.rake_screen_width_cm_spin:
            self._speak_auto_text(self.rake_screen_width_cm_spin.text())
        elif sender is self.rake_screen_height_cm_spin:
            self._speak_auto_text(self.rake_screen_height_cm_spin.text())
        elif sender is self.rake_spacing_spin:
            self._speak_auto_text(self.rake_spacing_spin.text())
        elif sender is self.rake_gaze_smoothing_spin:
            self._speak_auto_text(self.rake_gaze_smoothing_spin.text())
        elif sender is self.rake_gaze_gain_x_spin:
            self._speak_auto_text(self.rake_gaze_gain_x_spin.text())
        elif sender is self.rake_gaze_gain_y_spin:
            self._speak_auto_text(self.rake_gaze_gain_y_spin.text())
        elif sender is self.rake_gaze_offset_x_spin:
            self._speak_auto_text(self.rake_gaze_offset_x_spin.text())
        elif sender is self.rake_gaze_offset_y_spin:
            self._speak_auto_text(self.rake_gaze_offset_y_spin.text())
        elif sender is self.rake_selection_hold_spin:
            self._speak_auto_text(self.rake_selection_hold_spin.text())
        elif sender is self.rake_calib_points_combo:
            self._speak_auto_text(self.rake_calib_points_combo.currentText())
        elif sender is self.display_cb:
            key = "turn_on" if self.display_cb.isChecked() else "turn_off"
            self._speak_control_name(self._format_text(key, name=self._text("display_short")))
        elif sender is self.disable_accel_cb:
            key = "turn_on" if self.disable_accel_cb.isChecked() else "turn_off"
            self._speak_control_name(self._format_text(key, name=self._text("disable_accel_short")))
        elif sender is self.log_data_cb:
            key = "turn_on" if self.log_data_cb.isChecked() else "turn_off"
            self._speak_control_name(self._format_text(key, name=self._text("record_data")))
        elif sender is self.rake_lock_on_dwell_cb:
            key = "turn_on" if self.rake_lock_on_dwell_cb.isChecked() else "turn_off"
            self._speak_control_name(self._format_text(key, name=self._text("rake_lock_on_dwell")))
        elif sender is self.rake_show_gaze_cb:
            key = "turn_on" if self.rake_show_gaze_cb.isChecked() else "turn_off"
            self._speak_control_name(self._format_text(key, name=self._text("rake_show_gaze")))
        elif sender is self.rake_without_targetfinder_cb:
            key = "turn_on" if self.rake_without_targetfinder_cb.isChecked() else "turn_off"
            self._speak_control_name(self._format_text(key, name=self._text("rake_without_targetfinder")))
        elif sender is self.experiment_enabled_cb:
            key = "turn_on" if self.experiment_enabled_cb.isChecked() else "turn_off"
            self._speak_control_name(self._format_text(key, name=self._text("experiment_enabled")))
        elif sender is self.experiment_fullscreen_cb:
            key = "turn_on" if self.experiment_fullscreen_cb.isChecked() else "turn_off"
            self._speak_control_name(self._format_text(key, name=self._text("experiment_fullscreen")))
        elif sender is self.experiment_show_all_targets_cb:
            key = "turn_on" if self.experiment_show_all_targets_cb.isChecked() else "turn_off"
            self._speak_control_name(self._format_text(key, name=self._text("experiment_show_all_targets")))

    def _handle_rake_calibration_toggle(self, checked: bool):
        if self._suspend_updates:
            return
        if not checked:
            self._rake_calibration_status = "not_calibrated"
            self._rake_calibration_status_detail = None
        self._update_rake_calibration_ui()
        self._handle_runtime_option_change()
        key = "turn_on" if checked else "turn_off"
        self._speak_control_name(self._format_text(key, name=self._text("rake_use_calibration")))

    def _handle_rake_reset_calibration(self):
        self.rake_use_calibration_cb.setChecked(False)
        self._rake_calibration_status = "not_calibrated"
        self._rake_calibration_status_detail = None
        self._update_rake_calibration_ui()
        cfg = self._save_config()
        if self._is_demo_running() and self._mode_code() == "rake":
            self._stop_demo(silent=True)
            self._launch_demo_for_config(cfg, speak=False)

    def _handle_numeric_edit_finished(self, widget):
        if self._suspend_updates:
            return
        widget.interpretText()
        self._save_config()
        self._set_status("pending_apply")
        self._speak_auto_text(widget.text())

    def _handle_apply_clicked(self):
        focus_widget = self.focusWidget()
        if focus_widget is not None:
            focus_widget.clearFocus()
        QtWidgets.QApplication.processEvents()
        self._commit_numeric_inputs()
        cfg = self._save_config()
        if self._mode_code() is None:
            self._set_status("select_mode_first")
            return
        if cfg.enable_rake_cursor and cfg.rake_use_calibration:
            cfg.rake_auto_calibrate = True
            self._rake_calibration_status = "calibrating"
            self._rake_calibration_status_detail = None
            self._update_rake_calibration_ui()
        if self._is_demo_running():
            self._stop_demo(silent=True)
        self._launch_demo_for_config(cfg, speak=True)

    def _handle_panel_style_change(self, *_args):
        if self._suspend_updates:
            return
        self._apply_panel_style()
        self._save_config()
        self._set_status("panel_updated")
        key = "turn_on" if self.high_contrast_cb.isChecked() else "turn_off"
        self._speak_control_name(self._format_text(key, name=self._text("contrast")))

    def _handle_tts_change(self, checked: bool):
        if self._suspend_updates:
            return
        self._save_config()
        if checked and not self._tts_available():
            self.enable_tts_cb.blockSignals(True)
            self.enable_tts_cb.setChecked(False)
            self.enable_tts_cb.blockSignals(False)
            self._save_config()
            self._set_status("tts_unavailable")
            return
        self._set_status("tts_enabled" if checked else "tts_disabled")
        if checked:
            self._speak_control_name(self._text("enable_tts"))

    def _handle_language_selection(self):
        self._speak_control_name(self._text("choose_language_dialog"))
        options = [
            ("English", self._language_label("English")),
            ("French", self._language_label("French")),
        ]
        selected = self._show_selection_dialog(
            self._text("choose_language_dialog"),
            options,
            self._selected_language,
        )
        if selected is None:
            return
        self._selected_language = selected
        self._apply_language()
        self._save_config()
        self._set_status("language_updated")

    def _handle_model_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self._text("model_path"),
            str(self.project_root),
            "PyTorch model (*.pt)",
        )
        if not path:
            return
        self.model_path_edit.setText(path)
        self._save_config()
        self._set_status("pending_apply")

    def _handle_experiment_data_browse(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self._text("experiment_data_dir"),
            self.experiment_data_path_edit.text().strip() or DEFAULT_EXPERIMENT_DATA_DIR,
        )
        if not path:
            return
        self.experiment_data_path_edit.setText(path)
        self._save_config()
        self._set_status("pending_apply")

    def _apply_panel_style(self):
        bg_main = "#ececef"
        bg_sidebar = "#f4f4f7"
        bg_nav = "#f4f4f7"
        bg_nav_checked = "#e03a8a"
        bg_nav_hover = "#ececf2"
        bg_card = "#f7f7f9"

        text_main = "#1d1d1f"
        text_muted = "#667085"
        text_on_accent = "#ffffff"

        border_soft = "#d8d8de"
        border_card = "#cfcfd6"
        separator = "#d9d9e2"

        tool_btn_bg = "#f7f7f9"
        tool_btn_hover = "#ececf2"
        action_btn_bg = "#ffffff"
        action_btn_hover = "#f7f7f9"
        input_bg = "#ffffff"
        switch_off_bg = "#d1d1d6"
        switch_off_border = "#c7c7cc"
        switch_on_bg = "#34c759"

        if self.high_contrast_cb.isChecked():
            bg_main = "#ffffff"
            bg_sidebar = "#ffffff"
            bg_nav = "#ffffff"
            bg_nav_checked = "#111111"
            bg_nav_hover = "#f2f2f2"
            bg_card = "#ffffff"

            text_main = "#000000"
            text_muted = "#222222"
            text_on_accent = "#ffffff"

            border_soft = "#000000"
            border_card = "#000000"
            separator = "#000000"

            tool_btn_bg = "#ffffff"
            tool_btn_hover = "#f2f2f2"
            action_btn_bg = "#ffffff"
            action_btn_hover = "#f2f2f2"
            input_bg = "#ffffff"
            switch_off_bg = "#ffffff"
            switch_off_border = "#000000"
            switch_on_bg = "#111111"

        self.setStyleSheet(
            f"""
            QWidget {{
                background: {bg_main};
                color: {text_main};
                font-size: 14px;
            }}

            QScrollArea, QScrollArea > QWidget > QWidget {{
                background: transparent;
            }}

            QFrame#SidebarFrame {{
                background: {bg_sidebar};
                border: 1px solid {border_soft};
                border-radius: 22px;
            }}

            QPushButton#NavButton {{
                background: {bg_nav};
                border: none;
                border-bottom: 1px solid {border_soft};
                text-align: left;
                padding: 16px 14px;
                font-size: 17px;
                font-weight: 600;
                color: {text_main};
            }}

            QPushButton#NavButton[navRole="top"] {{
                border-top-left-radius: 22px;
                border-top-right-radius: 22px;
            }}

            QPushButton#NavButton[navRole="bottom"] {{
                border-bottom: none;
                border-bottom-left-radius: 22px;
                border-bottom-right-radius: 22px;
            }}

            QPushButton#NavButton:checked {{
                background: {bg_nav_checked};
                color: {text_on_accent};
            }}

            QPushButton#NavButton:hover:!checked {{
                background: {bg_nav_hover};
            }}

            QLabel#PageTitle {{
                font-size: 24px;
                font-weight: 700;
                background: transparent;
            }}

            QLabel#GroupTitle {{
                font-size: 16px;
                font-weight: 700;
                color: {text_main};
                background: transparent;
                padding: 4px 2px 10px 2px;
            }}

            QLabel#SettingLabel {{
                font-size: 14px;
                font-weight: 600;
                background: transparent;
            }}

            QLabel#SettingHelp {{
                font-size: 12px;
                color: {text_muted};
                background: transparent;
            }}

            QLabel#SettingValueLabel {{
                font-size: 14px;
                font-weight: 600;
                color: {text_main};
                background: transparent;
            }}

            QWidget#LabelColumn {{
                background: transparent;
            }}

            QLabel#InfoLabel {{
                background: transparent;
                font-size: 14px;
                color: {text_muted};
                padding: 2px 2px 0 2px;
            }}

            QLabel#SectionNote {{
                background: transparent;
                font-size: 13px;
                color: {text_muted};
                padding-top: 16px;
            }}

            QFrame#Card {{
                background: {bg_card};
                border: 2px solid {border_card};
                border-radius: 24px;
            }}

            QFrame#SettingGroup {{
                background: {bg_card};
                border: 2px solid {border_card};
                border-radius: 22px;
            }}

            QWidget#SettingRow {{
                background: transparent;
            }}

            QFrame#Separator {{
                background: {separator};
                max-height: 1px;
                border: none;
            }}

            QToolButton#HeaderNavButton {{
                background: {tool_btn_bg};
                border: 2px solid {border_card};
                border-radius: 21px;
                padding: 0;
                font-size: 22px;
                font-weight: 700;
                text-align: center;
            }}

            QToolButton#HeaderNavButton:hover {{
                background: {tool_btn_hover};
            }}

            QToolButton#HeaderNavButton:disabled {{
                color: {text_muted};
            }}

            QPushButton#ActionButton {{
                background: {action_btn_bg};
                border: 2px solid {border_card};
                border-radius: 12px;
                min-height: 40px;
                padding: 6px 16px;
                font-size: 15px;
                font-weight: 600;
            }}

            QPushButton#ActionButton:hover {{
                background: {action_btn_hover};
            }}

            QPushButton#ActionButton:disabled {{
                background: #f1f2f5;
                border-color: {border_soft};
                color: {text_muted};
            }}

            QPushButton#SelectorButton {{
                background: {input_bg};
                border: 1px solid {border_soft};
                border-radius: 10px;
                min-height: 38px;
                font-size: 15px;
                padding: 4px 12px;
                text-align: left;
            }}

            QPushButton#SelectorButton:hover {{
                background: {action_btn_hover};
            }}

            QLineEdit {{
                background: {input_bg};
                border: 1px solid {border_soft};
                border-radius: 10px;
                min-height: 34px;
                font-size: 14px;
                padding: 2px 8px;
                selection-background-color: {bg_nav_checked};
            }}

            QLineEdit:read-only {{
                background: {input_bg};
                color: {text_main};
            }}

            QWidget#ModelPicker {{
                background: transparent;
            }}

            QPushButton#SmallActionButton {{
                background: {action_btn_bg};
                border: 1px solid {border_soft};
                border-radius: 10px;
                min-height: 34px;
                padding: 4px 10px;
                font-size: 13px;
                font-weight: 600;
            }}

            QPushButton#SmallActionButton:hover {{
                background: {action_btn_hover};
            }}

            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {input_bg};
                border: 1px solid {border_soft};
                border-radius: 10px;
                min-height: 34px;
                font-size: 14px;
                padding: 2px 8px;
            }}

            QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
                background: #f1f2f5;
                color: {text_muted};
            }}

            QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 24px;
                border: none;
                background: transparent;
            }}

            QComboBox QAbstractItemView {{
                background: {input_bg};
                border: 1px solid {border_soft};
                selection-background-color: {bg_nav_checked};
                selection-color: {text_on_accent};
                font-size: 14px;
                padding: 4px;
                outline: 0;
            }}

            QCheckBox {{
                background: transparent;
            }}

            QCheckBox::indicator {{
                width: 46px;
                height: 26px;
                border-radius: 13px;
                background: {switch_off_bg};
                border: 1px solid {switch_off_border};
            }}

            QCheckBox::indicator:checked {{
                background: {switch_on_bg};
                border: 1px solid {switch_on_bg};
            }}

            QCheckBox::indicator:disabled {{
                background: #e5e7eb;
                border: 1px solid {border_soft};
            }}

            QLabel#DialogTitle {{
                font-size: 16px;
                font-weight: 700;
                background: transparent;
            }}

            QListWidget#SelectorList {{
                border: 1px solid {border_soft};
                border-radius: 12px;
                background: {input_bg};
                font-size: 14px;
                padding: 6px;
            }}

            QListWidget#SelectorList::item {{
                padding: 10px 12px;
                border-radius: 8px;
            }}

            QListWidget#SelectorList::item:selected {{
                background: {bg_nav_checked};
                color: {text_on_accent};
            }}
            """
        )

    # ---------------------
    # Config
    # ---------------------
    def _commit_numeric_inputs(self):
        for widget in (
            self.change_thresh_spin,
            self.capture_interval_spin,
            self.confidence_spin,
            self.iou_spin,
            self.filter_freq_spin,
            self.filter_min_cutoff_spin,
            self.filter_beta_spin,
            self.filter_d_cutoff_spin,
            self.dynaspot_min_speed_spin,
            self.dynaspot_spot_width_spin,
            self.dynaspot_lag_spin,
            self.dynaspot_reduce_time_spin,
            self.rake_camera_index_spin,
            self.rake_screen_width_cm_spin,
            self.rake_screen_height_cm_spin,
            self.rake_spacing_spin,
            self.rake_gaze_smoothing_spin,
            self.rake_gaze_gain_x_spin,
            self.rake_gaze_gain_y_spin,
            self.rake_gaze_offset_x_spin,
            self.rake_gaze_offset_y_spin,
            self.rake_selection_hold_spin,
            self.experiment_trials_spin,
            self.experiment_countdown_spin,
            self.experiment_max_clicks_spin,
        ):
            widget.interpretText()

    def _collect_config(self) -> PanelConfig:
        self._commit_numeric_inputs()
        mode = self._mode_code()
        return PanelConfig(
            ignore_text=self._hidden_config.ignore_text,
            ignore_large_targets=self._hidden_config.ignore_large_targets,
            show_bounding_boxes=self._hidden_config.show_bounding_boxes,
            show_class_labels=self._hidden_config.show_class_labels,
            confidence=self.confidence_spin.value(),
            change_thresh=self.change_thresh_spin.value(),
            capture_interval=self.capture_interval_spin.value(),
            iou=self.iou_spin.value(),
            model_path=self.model_path_edit.text().strip(),
            filter_name=self._selected_filter,
            filter_freq=self.filter_freq_spin.value(),
            filter_min_cutoff=self.filter_min_cutoff_spin.value(),
            filter_beta=self.filter_beta_spin.value(),
            filter_d_cutoff=self.filter_d_cutoff_spin.value(),
            enable_logging=self.log_data_cb.isChecked(),
            display=self.display_cb.isChecked(),
            disable_accel=self.disable_accel_cb.isChecked(),
            dynaspot_min_speed=self.dynaspot_min_speed_spin.value(),
            dynaspot_spot_width=self.dynaspot_spot_width_spin.value(),
            dynaspot_lag=self.dynaspot_lag_spin.value(),
            dynaspot_reduce_time=self.dynaspot_reduce_time_spin.value(),
            rake_camera_index=self.rake_camera_index_spin.value(),
            rake_screen_width_cm=self.rake_screen_width_cm_spin.value(),
            rake_screen_height_cm=self.rake_screen_height_cm_spin.value(),
            rake_spacing=self.rake_spacing_spin.value(),
            rake_gaze_smoothing=self.rake_gaze_smoothing_spin.value(),
            rake_gaze_gain_x=self.rake_gaze_gain_x_spin.value(),
            rake_gaze_gain_y=self.rake_gaze_gain_y_spin.value(),
            rake_gaze_offset_x=self.rake_gaze_offset_x_spin.value(),
            rake_gaze_offset_y=self.rake_gaze_offset_y_spin.value(),
            rake_selection_hold=self.rake_selection_hold_spin.value(),
            rake_lock_on_dwell=self.rake_lock_on_dwell_cb.isChecked(),
            rake_show_gaze=self.rake_show_gaze_cb.isChecked(),
            rake_without_targetfinder=self.rake_without_targetfinder_cb.isChecked(),
            rake_use_calibration=self.rake_use_calibration_cb.isChecked(),
            rake_calib_points=int(self.rake_calib_points_combo.currentText()),
            rake_auto_calibrate=False,
            rake_calibration_status=self._rake_calibration_status,
            experiment_enabled=self.experiment_enabled_cb.isChecked(),
            experiment_data_dir=self.experiment_data_path_edit.text().strip() or DEFAULT_EXPERIMENT_DATA_DIR,
            experiment_trials=self.experiment_trials_spin.value(),
            experiment_difficulty=self.experiment_difficulty_combo.currentText(),
            experiment_countdown=self.experiment_countdown_spin.value(),
            experiment_max_clicks=self.experiment_max_clicks_spin.value(),
            experiment_fullscreen=self.experiment_fullscreen_cb.isChecked(),
            experiment_show_all_targets=self.experiment_show_all_targets_cb.isChecked(),
            enable_bubble_cursor=mode == "bubble",
            enable_semantic_pointing=mode == "semantic",
            enable_dynaspot=mode == "dynaspot",
            enable_rake_cursor=mode == "rake",
            high_contrast_mode=self.high_contrast_cb.isChecked(),
            stronger_visual_cue=self._hidden_config.stronger_visual_cue,
            single_click_as_double_click=self._hidden_config.single_click_as_double_click,
            preset="TargetFinder" if mode == "targetfinder" else "Bubble Only" if mode == "bubble" else "Semantic Only" if mode == "semantic" else "DynaSpot" if mode == "dynaspot" else "Ninja Cursors(gaze)" if mode == "rake" else "",
            enable_tts=self.enable_tts_cb.isChecked(),
            language=self._language_code(),
        )

    def _apply_config(self, cfg: PanelConfig):
        self._hidden_config = cfg
        self.change_thresh_spin.setValue(cfg.change_thresh)
        self.capture_interval_spin.setValue(cfg.capture_interval)
        self.confidence_spin.setValue(cfg.confidence)
        self.iou_spin.setValue(cfg.iou)
        self.model_path_edit.setText(cfg.model_path)
        self._selected_filter = cfg.filter_name if cfg.filter_name in FILTER_OPTIONS else "none"
        self.filter_freq_spin.setValue(cfg.filter_freq)
        self.filter_min_cutoff_spin.setValue(cfg.filter_min_cutoff)
        self.filter_beta_spin.setValue(cfg.filter_beta)
        self.filter_d_cutoff_spin.setValue(cfg.filter_d_cutoff)
        self.log_data_cb.setChecked(cfg.enable_logging)
        self.display_cb.setChecked(cfg.display)
        self.disable_accel_cb.setChecked(cfg.disable_accel)
        self.dynaspot_min_speed_spin.setValue(cfg.dynaspot_min_speed)
        self.dynaspot_spot_width_spin.setValue(cfg.dynaspot_spot_width)
        self.dynaspot_lag_spin.setValue(cfg.dynaspot_lag)
        self.dynaspot_reduce_time_spin.setValue(cfg.dynaspot_reduce_time)
        self.rake_camera_index_spin.setValue(cfg.rake_camera_index)
        self.rake_screen_width_cm_spin.setValue(cfg.rake_screen_width_cm)
        self.rake_screen_height_cm_spin.setValue(cfg.rake_screen_height_cm)
        self.rake_spacing_spin.setValue(cfg.rake_spacing)
        self.rake_gaze_smoothing_spin.setValue(cfg.rake_gaze_smoothing)
        self.rake_gaze_gain_x_spin.setValue(cfg.rake_gaze_gain_x)
        self.rake_gaze_gain_y_spin.setValue(cfg.rake_gaze_gain_y)
        self.rake_gaze_offset_x_spin.setValue(cfg.rake_gaze_offset_x)
        self.rake_gaze_offset_y_spin.setValue(cfg.rake_gaze_offset_y)
        self.rake_selection_hold_spin.setValue(cfg.rake_selection_hold)
        self.rake_lock_on_dwell_cb.setChecked(cfg.rake_lock_on_dwell)
        self.rake_show_gaze_cb.setChecked(cfg.rake_show_gaze)
        self.rake_without_targetfinder_cb.setChecked(cfg.rake_without_targetfinder)
        self.rake_use_calibration_cb.setChecked(cfg.rake_use_calibration)
        self.rake_calib_points_combo.setCurrentText(str(cfg.rake_calib_points))
        self._rake_calibration_status = cfg.rake_calibration_status or "not_calibrated"
        self._rake_calibration_status_detail = None
        self.experiment_enabled_cb.setChecked(cfg.experiment_enabled)
        self.experiment_data_path_edit.setText(cfg.experiment_data_dir or DEFAULT_EXPERIMENT_DATA_DIR)
        self.experiment_trials_spin.setValue(cfg.experiment_trials)
        self.experiment_difficulty_combo.setCurrentText(
            cfg.experiment_difficulty if cfg.experiment_difficulty in {"easy", "medium", "hard", "mixed"} else DEFAULT_EXPERIMENT_DIFFICULTY
        )
        self.experiment_countdown_spin.setValue(cfg.experiment_countdown)
        self.experiment_max_clicks_spin.setValue(cfg.experiment_max_clicks)
        self.experiment_fullscreen_cb.setChecked(cfg.experiment_fullscreen)
        self.experiment_show_all_targets_cb.setChecked(cfg.experiment_show_all_targets)
        self.high_contrast_cb.setChecked(cfg.high_contrast_mode)
        self.enable_tts_cb.setChecked(cfg.enable_tts)

        if cfg.preset == "TargetFinder":
            self._selected_mode = "targetfinder"
        elif cfg.enable_bubble_cursor or cfg.preset == "Bubble Only":
            self._selected_mode = "bubble"
        elif cfg.enable_semantic_pointing or cfg.preset == "Semantic Only":
            self._selected_mode = "semantic"
        elif cfg.enable_dynaspot or cfg.preset == "DynaSpot":
            self._selected_mode = "dynaspot"
        elif cfg.enable_rake_cursor or cfg.preset in {"Rake Cursor", "Ninja Cursors(gaze)"}:
            self._selected_mode = "rake"
        else:
            self._selected_mode = None

        self._selected_language = cfg.language if cfg.language in {"English", "French"} else "French"

        self._apply_language()
        self._apply_panel_style()
        self._refresh_filter_selector_text()
        self._update_mode_dependent_fields()

    def _save_config(self):
        cfg = self._collect_config()
        self.config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        self._hidden_config = cfg
        return cfg

    def _load_if_exists(self):
        if not self.config_path.exists():
            return

        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        valid_fields = {field.name for field in fields(PanelConfig)}
        filtered = {key: value for key, value in data.items() if key in valid_fields}
        cfg = PanelConfig(**filtered)
        cfg.change_thresh = DEFAULT_CHANGE_THRESH
        cfg.capture_interval = DEFAULT_CAPTURE_INTERVAL
        cfg.confidence = DEFAULT_CONFIDENCE
        cfg.iou = DEFAULT_IOU
        cfg.model_path = ""
        cfg.filter_name = "none"
        cfg.filter_freq = DEFAULT_FILTER_FREQ
        cfg.filter_min_cutoff = DEFAULT_FILTER_MIN_CUTOFF
        cfg.filter_beta = DEFAULT_FILTER_BETA
        cfg.filter_d_cutoff = DEFAULT_FILTER_D_CUTOFF
        cfg.enable_logging = False
        cfg.enable_bubble_cursor = False
        cfg.enable_semantic_pointing = False
        cfg.enable_dynaspot = False
        cfg.enable_rake_cursor = False
        cfg.display = False
        cfg.disable_accel = False
        cfg.dynaspot_min_speed = DEFAULT_DYNASPOT_MIN_SPEED
        cfg.dynaspot_spot_width = DEFAULT_DYNASPOT_SPOT_WIDTH
        cfg.dynaspot_lag = DEFAULT_DYNASPOT_LAG
        cfg.dynaspot_reduce_time = DEFAULT_DYNASPOT_REDUCE_TIME
        detected_screen_width_cm, detected_screen_height_cm = self._current_screen_physical_size_cm()
        cfg.rake_camera_index = DEFAULT_RAKE_CAMERA_INDEX
        cfg.rake_screen_width_cm = detected_screen_width_cm
        cfg.rake_screen_height_cm = detected_screen_height_cm
        cfg.rake_spacing = DEFAULT_RAKE_SPACING
        cfg.rake_gaze_smoothing = DEFAULT_RAKE_GAZE_SMOOTHING
        cfg.rake_gaze_gain_x = DEFAULT_RAKE_GAZE_GAIN_X
        cfg.rake_gaze_gain_y = DEFAULT_RAKE_GAZE_GAIN_Y
        cfg.rake_gaze_offset_x = DEFAULT_RAKE_GAZE_OFFSET_X
        cfg.rake_gaze_offset_y = DEFAULT_RAKE_GAZE_OFFSET_Y
        cfg.rake_selection_hold = DEFAULT_RAKE_SELECTION_HOLD
        cfg.rake_lock_on_dwell = DEFAULT_RAKE_LOCK_ON_DWELL
        cfg.rake_show_gaze = True
        cfg.rake_without_targetfinder = DEFAULT_RAKE_WITHOUT_TARGETFINDER
        cfg.rake_use_calibration = DEFAULT_RAKE_USE_CALIBRATION
        cfg.rake_calib_points = DEFAULT_RAKE_CALIB_POINTS
        cfg.rake_auto_calibrate = DEFAULT_RAKE_AUTO_CALIBRATE
        cfg.rake_calibration_status = "not_calibrated"
        cfg.experiment_enabled = False
        cfg.experiment_data_dir = DEFAULT_EXPERIMENT_DATA_DIR
        cfg.experiment_trials = DEFAULT_EXPERIMENT_TRIALS
        cfg.experiment_difficulty = DEFAULT_EXPERIMENT_DIFFICULTY
        cfg.experiment_countdown = DEFAULT_EXPERIMENT_COUNTDOWN
        cfg.experiment_max_clicks = DEFAULT_EXPERIMENT_MAX_CLICKS
        cfg.experiment_fullscreen = DEFAULT_EXPERIMENT_FULLSCREEN
        cfg.experiment_show_all_targets = DEFAULT_EXPERIMENT_SHOW_ALL_TARGETS
        cfg.high_contrast_mode = False
        cfg.enable_tts = False
        cfg.language = "French"
        cfg.preset = ""
        self._apply_config(cfg)

    # ---------------------------
    # Command / Process
    # ---------------------------
    def _build_command(self, cfg: PanelConfig):
        if cfg.experiment_enabled:
            return self._build_experiment_command(cfg)
        if cfg.preset == "TargetFinder":
            module_name = "target_finder_toolkit.targetfinder"
        elif cfg.enable_bubble_cursor:
            module_name = "target_finder_toolkit.bubblecursor"
        elif cfg.enable_dynaspot:
            module_name = "target_finder_toolkit.dynaspot"
        elif cfg.enable_rake_cursor:
            module_name = "target_finder_toolkit.ninjacursors"
        else:
            module_name = "target_finder_toolkit.semanticpointing"
        cmd = [sys.executable, "-m", module_name]
        cmd += ["--change-thresh", str(cfg.change_thresh)]
        cmd += ["--capture-interval", str(cfg.capture_interval)]
        cmd += ["--confidence", str(cfg.confidence)]
        cmd += ["--iou", str(cfg.iou)]
        cmd += ["--filter", cfg.filter_name]
        cmd += [
            "--filter-freq", str(cfg.filter_freq),
            "--filter-min-cutoff", str(cfg.filter_min_cutoff),
            "--filter-beta", str(cfg.filter_beta),
            "--filter-d-cutoff", str(cfg.filter_d_cutoff),
        ]
        if cfg.model_path:
            cmd += ["--model-path", cfg.model_path]
        if cfg.enable_logging:
            mode_name = cfg.preset or self._mode_code() or "session"
            log_path = make_default_log_path(self.project_root, mode_name)
            cmd += ["--log-file", str(log_path), "--log-cursor-hz", "30"]

        if cfg.enable_semantic_pointing and cfg.display:
            cmd.append("--display")
        if cfg.enable_semantic_pointing and cfg.disable_accel:
            cmd.append("--disable-accel")
        if cfg.enable_dynaspot:
            cmd += [
                "--min-speed", str(cfg.dynaspot_min_speed),
                "--spot-width", str(cfg.dynaspot_spot_width),
                "--lag", str(cfg.dynaspot_lag),
                "--reduce-time", str(cfg.dynaspot_reduce_time),
            ]
        if cfg.enable_rake_cursor:
            cmd += [
                "--camera-index", str(cfg.rake_camera_index),
                "--screen-width-cm", str(cfg.rake_screen_width_cm),
                "--screen-height-cm", str(cfg.rake_screen_height_cm),
                "--ninja-spacing", str(cfg.rake_spacing),
                "--gaze-smoothing", str(cfg.rake_gaze_smoothing),
                "--gaze-gain-x", str(cfg.rake_gaze_gain_x),
                "--gaze-gain-y", str(cfg.rake_gaze_gain_y),
                "--gaze-offset-x", str(cfg.rake_gaze_offset_x),
                "--gaze-offset-y", str(cfg.rake_gaze_offset_y),
                "--selection-hold", str(cfg.rake_selection_hold),
            ]
            if cfg.rake_lock_on_dwell:
                cmd.append("--lock-on-dwell")
            if cfg.rake_use_calibration:
                cmd += ["--calib-points", str(cfg.rake_calib_points)]
                if cfg.rake_auto_calibrate:
                    cmd.append("--auto-calibrate")
            if not cfg.rake_show_gaze:
                cmd.append("--hide-gaze-point")
            if cfg.rake_without_targetfinder:
                cmd.append("--without-targetfinder")
        return cmd

    def _experiment_technique_for_config(self, cfg: PanelConfig) -> str:
        if cfg.preset == "TargetFinder":
            return "mouse"
        if cfg.enable_bubble_cursor:
            return "bubble"
        if cfg.enable_dynaspot:
            return "dynaspot"
        if cfg.enable_rake_cursor:
            return "ninja_cursors"
        return "semantic"

    def _build_experiment_command(self, cfg: PanelConfig):
        technique = self._experiment_technique_for_config(cfg)
        cmd = [
            sys.executable,
            "-m",
            "target_finder_toolkit.experimental_task",
            "--technique",
            technique,
            "--data-dir",
            cfg.experiment_data_dir,
            "--trials",
            str(cfg.experiment_trials),
            "--difficulty",
            cfg.experiment_difficulty,
            "--countdown",
            str(cfg.experiment_countdown),
            "--max-clicks",
            str(cfg.experiment_max_clicks),
            "--change-thresh",
            str(cfg.change_thresh),
            "--capture-interval",
            str(cfg.capture_interval),
            "--confidence",
            str(cfg.confidence),
            "--iou",
            str(cfg.iou),
            "--filter",
            cfg.filter_name,
            "--filter-freq",
            str(cfg.filter_freq),
            "--filter-min-cutoff",
            str(cfg.filter_min_cutoff),
            "--filter-beta",
            str(cfg.filter_beta),
            "--filter-d-cutoff",
            str(cfg.filter_d_cutoff),
        ]
        if cfg.model_path:
            cmd += ["--model-path", cfg.model_path]
        if not cfg.experiment_fullscreen:
            cmd.append("--windowed")
        if cfg.experiment_show_all_targets:
            cmd.append("--show-all-targets")
        if not cfg.enable_logging:
            cmd.append("--no-technique-log")
        if cfg.enable_semantic_pointing and cfg.display:
            cmd.append("--semantic-display")
        if cfg.enable_semantic_pointing and cfg.disable_accel:
            cmd.append("--semantic-disable-accel")
        if cfg.enable_dynaspot:
            cmd += [
                "--dynaspot-min-speed", str(cfg.dynaspot_min_speed),
                "--dynaspot-spot-width", str(cfg.dynaspot_spot_width),
                "--dynaspot-lag", str(cfg.dynaspot_lag),
                "--dynaspot-reduce-time", str(cfg.dynaspot_reduce_time),
            ]
        if cfg.enable_rake_cursor:
            cmd += [
                "--ninja-camera-index", str(cfg.rake_camera_index),
                "--ninja-screen-width-cm", str(cfg.rake_screen_width_cm),
                "--ninja-screen-height-cm", str(cfg.rake_screen_height_cm),
                "--ninja-spacing", str(cfg.rake_spacing),
                "--ninja-gaze-smoothing", str(cfg.rake_gaze_smoothing),
                "--ninja-gaze-gain-x", str(cfg.rake_gaze_gain_x),
                "--ninja-gaze-gain-y", str(cfg.rake_gaze_gain_y),
                "--ninja-gaze-offset-x", str(cfg.rake_gaze_offset_x),
                "--ninja-gaze-offset-y", str(cfg.rake_gaze_offset_y),
                "--ninja-selection-hold", str(cfg.rake_selection_hold),
                "--ninja-calib-points", str(cfg.rake_calib_points),
            ]
            if cfg.rake_lock_on_dwell:
                cmd.append("--ninja-lock-on-dwell")
            if not cfg.rake_show_gaze:
                cmd.append("--ninja-hide-gaze-point")
            if cfg.rake_use_calibration:
                cmd.append("--ninja-auto-calibrate")
            if not cfg.rake_without_targetfinder:
                cmd.append("--ninja-with-targetfinder")
        return cmd

    def _is_demo_running(self):
        return self.process is not None and self.process.poll() is None

    def _handle_process_output_line(self, line: str):
        prefix = "__NINJA_CALIB__ "
        if not line.startswith(prefix):
            return
        try:
            payload = json.loads(line[len(prefix):])
        except Exception:
            return
        event = payload.get("event")
        if event == "started":
            self._rake_calibration_status = "calibrating"
            points = payload.get("num_points")
            self._rake_calibration_status_detail = f"{points} pts" if points else None
        elif event == "calibrated":
            self._rake_calibration_status = "calibrated"
            mean_error_px = payload.get("mean_error_px")
            self._rake_calibration_status_detail = (
                f"{float(mean_error_px):.0f}px"
                if mean_error_px is not None
                else None
            )
            correction_values = payload.get("correction_values") or {}
            self._apply_calibration_correction_values(correction_values)
        elif event == "failed":
            self._rake_calibration_status = "failed"
            self._rake_calibration_status_detail = None
        elif event == "cancelled":
            self._rake_calibration_status = "cancelled"
            self._rake_calibration_status_detail = None
        else:
            return
        self._update_rake_calibration_ui()

    def _apply_calibration_correction_values(self, values):
        if not isinstance(values, dict):
            return
        updates = [
            (self.rake_gaze_gain_x_spin, values.get("gaze_gain_x")),
            (self.rake_gaze_gain_y_spin, values.get("gaze_gain_y")),
            (self.rake_gaze_offset_x_spin, values.get("gaze_offset_x")),
            (self.rake_gaze_offset_y_spin, values.get("gaze_offset_y")),
        ]
        changed = False
        self._suspend_updates = True
        try:
            for widget, value in updates:
                if value is None:
                    continue
                try:
                    widget.setValue(float(value))
                    changed = True
                except (TypeError, ValueError):
                    continue
        finally:
            self._suspend_updates = False
        if changed:
            self._save_config()

    def _drain_process_output(self):
        if self.process is None or self.process.stdout is None:
            return
        fd = self.process.stdout.fileno()
        chunks = []
        while True:
            try:
                data = os.read(fd, 65536)
            except BlockingIOError:
                break
            except OSError:
                break
            if not data:
                break
            chunks.append(data.decode("utf-8", errors="replace"))
        if not chunks:
            return
        self._process_output_buffer += "".join(chunks)
        while True:
            newline_idx = self._process_output_buffer.find("\n")
            if newline_idx < 0:
                break
            line = self._process_output_buffer[:newline_idx].rstrip("\r")
            self._process_output_buffer = self._process_output_buffer[newline_idx + 1:]
            if line:
                self._process_output_lines.append(line)
                self._process_output_lines = self._process_output_lines[-20:]
            self._handle_process_output_line(line)

    def _format_process_error(self, exit_code: int) -> str:
        tail = "\n".join(self._process_output_lines[-8:]).strip()
        if not tail:
            return f"{self._text('stopped')} (exit code {exit_code})"
        return f"{self._text('stopped')} (exit code {exit_code})\n{tail}"

    def _poll_process_state(self):
        if self.process is None:
            self._process_watch_timer.stop()
            self._update_action_buttons()
            return
        self._drain_process_output()
        if self.process.poll() is None:
            return
        self._drain_process_output()
        exit_code = self.process.poll()
        if self._process_output_buffer.strip():
            self._process_output_lines.append(self._process_output_buffer.strip())
            self._process_output_lines = self._process_output_lines[-20:]
        self.process = None
        self._process_output_buffer = ""
        self._process_watch_timer.stop()
        restore_default_cursors()
        self._update_action_buttons()
        if exit_code:
            self.info_label.setText(self._format_process_error(exit_code))
        else:
            self._set_status("stopped", speak=False)

    def _launch_demo_for_config(self, cfg: PanelConfig, *, speak: bool):
        if cfg.experiment_enabled and not Path(cfg.experiment_data_dir).is_dir():
            self._set_status("invalid_experiment_data_dir", speak=speak)
            return
        uses_model = (
            not cfg.experiment_enabled
            or cfg.enable_semantic_pointing
            or cfg.enable_dynaspot
            or (cfg.enable_rake_cursor and not cfg.rake_without_targetfinder)
        )
        if uses_model and cfg.model_path and not Path(cfg.model_path).is_file():
            self._set_status("invalid_model_path", speak=speak)
            return
        if cfg.filter_name == "one_euro":
            try:
                importlib.import_module("OneEuroFilter")
            except Exception as exc:
                self.info_label.setText(f"{self._text('missing_one_euro')} ({exc})")
                return
        if cfg.enable_rake_cursor:
            try:
                _ensure_mediapipe_python_alias()
                importlib.import_module("webeyetrack")
            except Exception as exc:
                self.info_label.setText(f"{self._text('missing_webeyetrack')} ({exc})")
                return
        cmd = self._build_command(cfg)
        popen_kwargs = {
            "cwd": str(self.project_root),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
        }
        if sys.platform.startswith("win"):
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        self.process = subprocess.Popen(cmd, **popen_kwargs)
        self._process_output_buffer = ""
        self._process_output_lines = []
        if self.process.stdout is not None:
            try:
                os.set_blocking(self.process.stdout.fileno(), False)
            except Exception:
                pass
        self._process_watch_timer.start()
        self._update_action_buttons()
        if cfg.experiment_enabled:
            self._set_status("running_experiment", speak=speak)
        elif cfg.preset == "TargetFinder":
            self._set_status("running_targetfinder", speak=speak)
        elif cfg.enable_bubble_cursor:
            self._set_status("running_bubble", speak=speak)
        elif cfg.enable_dynaspot:
            self._set_status("running_dynaspot", speak=speak)
        elif cfg.enable_rake_cursor:
            self._set_status("running_rake", speak=speak)
        else:
            self._set_status("running_semantic", speak=speak)

    def _stop_demo(self, silent: bool = False):
        if not self._is_demo_running():
            self.process = None
            self._process_watch_timer.stop()
            restore_default_cursors()
            self._update_action_buttons()
            if not silent:
                self._set_status("no_running", speak=False)
            return

        try:
            if sys.platform.startswith("win"):
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.send_signal(signal.SIGTERM)
        except Exception:
            self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        finally:
            self._drain_process_output()
            self.process = None
            self._process_output_buffer = ""
            self._process_watch_timer.stop()
            restore_default_cursors()
            self._update_action_buttons()

        if not silent:
            self._set_status("stopped", speak=self.enable_tts_cb.isChecked())

    # ---------------------------
    # TTS
    # ---------------------------
    def _mac_voice_name(self):
        if sys.platform != "darwin" or not shutil.which("say"):
            return None
        if self._mac_voice_names is None:
            try:
                result = subprocess.run(
                    ["say", "-v", "?"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self._mac_voice_names = {
                    line.split(maxsplit=1)[0]
                    for line in result.stdout.splitlines()
                    if line.strip()
                }
            except OSError:
                self._mac_voice_names = set()
        preferred = {
            "French": ["Thomas", "Aurelie", "Amelie"],
            "English": ["Samantha", "Daniel", "Alex"],
        }
        for voice in preferred.get(self._language_code(), []):
            if voice in self._mac_voice_names:
                return voice
        return None

    def _tts_command(self, message: str):
        if sys.platform == "darwin" and shutil.which("say"):
            voice = self._mac_voice_name()
            if voice:
                return ["say", "-v", voice, message]
            return ["say", message]
        if sys.platform.startswith("linux"):
            if shutil.which("spd-say"):
                return ["spd-say", message]
            if shutil.which("espeak"):
                return ["espeak", message]
        if sys.platform.startswith("win") and shutil.which("powershell"):
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Speak($args[0])"
            )
            return ["powershell", "-NoProfile", "-Command", script, message]
        return None

    def _tts_available(self):
        return self._tts_command("test") is not None

    def _speak(self, message: str):
        if not self.enable_tts_cb.isChecked():
            return
        cleaned = (
            message.replace("🇬🇧", "")
            .replace("🇫🇷", "")
            .replace("▼", "")
            .replace("▾", "")
            .strip()
        )
        if not cleaned:
            return
        cmd = self._tts_command(cleaned)
        if not cmd:
            return
        try:
            if self._speech_process is not None and self._speech_process.poll() is None:
                self._speech_process.terminate()
            self._speech_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass

    def _stop_speech(self):
        if self._speech_process is None:
            return
        try:
            if self._speech_process.poll() is None:
                self._speech_process.terminate()
                self._speech_process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self._speech_process.kill()
            except OSError:
                pass
        finally:
            self._speech_process = None

    # ---------------------------
    # Events
    # ---------------------------
    def closeEvent(self, event):
        self._process_watch_timer.stop()
        self._stop_demo(silent=True)
        self._stop_speech()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ControlPanel()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
