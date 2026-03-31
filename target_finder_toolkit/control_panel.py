import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from PyQt6 import QtCore, QtWidgets


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
        "technique": "Technique (4 modes)",
        "select_technique": "Choose a technique",
        "choose_mode_dialog": "Choose a Technique",
        "mode_targetfinder": "TargetFinder Overlay",
        "mode_bubble": "Bubble Cursor",
        "mode_semantic": "Semantic Pointing",
        "mode_dynaspot": "DynaSpot",
        "apply": "Start / Apply",
        "change_thresh": "Change Threshold (range: 0-100000, default: 100)",
        "change_thresh_desc": "Higher = fewer refreshes for small screen changes. Lower = reacts sooner.",
        "model_path": "YOLO Model (.pt file)",
        "model_path_desc": "Leave empty to use the packaged best.pt. Choose another trained .pt file to switch models.",
        "browse": "Browse",
        "use_default_model": "Use packaged best.pt",
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
        "mode_note": "TargetFinder Overlay: shows detected boxes for testing. Bubble Cursor: expands selection around the nearest target. Semantic Pointing: slows pointer movement near targets for easier aiming. DynaSpot: grows a circular activation area as pointer speed increases, while staying point-like at low speed.",
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
        "stopped": "Stopped the running mode.",
        "no_running": "No running mode was found.",
        "invalid_model_path": "The selected model file was not found.",
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
        "technique": "Technique (4 modes)",
        "select_technique": "Choisir une technique",
        "choose_mode_dialog": "Choisir une technique",
        "mode_targetfinder": "Overlay TargetFinder",
        "mode_bubble": "Bubble Cursor",
        "mode_semantic": "Pointage sémantique",
        "mode_dynaspot": "DynaSpot",
        "apply": "Démarrer / Appliquer",
        "change_thresh": "Seuil de changement (plage : 0-100000, défaut : 100)",
        "change_thresh_desc": "Plus haut = moins de rafraîchissements pour de petits changements. Plus bas = réaction plus rapide.",
        "model_path": "Modèle YOLO (fichier .pt)",
        "model_path_desc": "Laissez vide pour utiliser le fichier best.pt intégré. Choisissez un autre fichier .pt entraîné pour changer de modèle.",
        "browse": "Parcourir",
        "use_default_model": "Utiliser le best.pt intégré",
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
        "mode_note": "Overlay TargetFinder : affiche les boîtes détectées pour les tests. Bubble Cursor : agrandit la sélection autour de la cible la plus proche. Pointage sémantique : ralentit le pointeur près des cibles pour mieux viser. DynaSpot : agrandit une zone d’activation circulaire quand la vitesse du pointeur augmente, tout en restant ponctuel à faible vitesse.",
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
        "stopped": "Le mode en cours a été arrêté.",
        "no_running": "Aucun mode en cours n'a été trouvé.",
        "invalid_model_path": "Le fichier du modèle sélectionné est introuvable.",
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
    display: bool = False
    disable_accel: bool = False

    enable_bubble_cursor: bool = False
    enable_semantic_pointing: bool = False
    enable_dynaspot: bool = False

    high_contrast_mode: bool = False
    stronger_visual_cue: bool = False
    single_click_as_double_click: bool = False

    preset: str = ""
    enable_tts: bool = False
    language: str = "English"


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
        self._selected_language = "English"
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_path = self.project_root / "control_panel_config.json"

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
        if hasattr(self, "language_selector_button"):
            self._refresh_language_selector_text()
        if hasattr(self, "model_path_edit"):
            self.model_path_edit.setPlaceholderText(self._text("use_default_model"))

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

    def _language_label(self, code: str):
        item = LANGUAGE_OPTIONS[code]
        return f"{item['badge']} {item[self._language_code()]}"

    def _refresh_mode_selector_text(self):
        self.mode_selector_button.setText(f"{self._mode_label(self._selected_mode)}  ▼")

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

        card, card_layout = self._create_card()

        self.mode_selector_button = QtWidgets.QPushButton()
        self.mode_selector_button.setObjectName("SelectorButton")
        self._refresh_mode_selector_text()

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

        self.display_cb = self._create_switch()
        self.disable_accel_cb = self._create_switch()

        rows = [
            self._create_field_row("model_path", self.model_picker, "model_path_desc"),
            self._create_separator(),
            self._create_field_row("technique", self.mode_selector_button, "mode_note"),
            self._create_separator(),
            self._create_field_row("change_thresh", self.change_thresh_spin, "change_thresh_desc"),
            self._create_separator(),
            self._create_field_row("capture_interval", self.capture_interval_spin, "capture_interval_desc"),
            self._create_separator(),
            self._create_field_row("confidence", self.confidence_spin, "confidence_desc"),
            self._create_separator(),
            self._create_field_row("iou", self.iou_spin, "iou_desc"),
            self._create_separator(),
            self._create_switch_row("display", self.display_cb, "display_desc"),
            self._create_separator(),
            self._create_switch_row("disable_accel", self.disable_accel_cb, "disable_accel_desc"),
        ]

        for row in rows:
            card_layout.addWidget(row)

        page_layout.addWidget(card)
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
        self.language_selector_button.clicked.connect(self._handle_language_selection)
        self.model_browse_button.clicked.connect(self._handle_model_browse)
        self.start_button.clicked.connect(self._handle_apply_clicked)
        self.stop_button.clicked.connect(self._stop_demo)
        self.change_thresh_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.capture_interval_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.confidence_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.iou_spin.valueChanged.connect(self._handle_runtime_option_change)
        self.display_cb.toggled.connect(self._handle_runtime_option_change)
        self.disable_accel_cb.toggled.connect(self._handle_runtime_option_change)

        self.high_contrast_cb.toggled.connect(self._handle_panel_style_change)
        self.enable_tts_cb.toggled.connect(self._handle_tts_change)

        self._register_numeric_field(self.change_thresh_spin, "change_thresh")
        self._register_numeric_field(self.capture_interval_spin, "capture_interval")
        self._register_numeric_field(self.confidence_spin, "confidence")
        self._register_numeric_field(self.iou_spin, "iou")
        self._register_help_targets(
            [self.model_picker, self.model_path_edit, self.model_browse_button],
            "model_path",
            "model_path_desc",
        )

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

    def _update_mode_dependent_fields(self):
        semantic_enabled = self._mode_code() == "semantic"
        self.display_cb.setEnabled(semantic_enabled)
        self.disable_accel_cb.setEnabled(semantic_enabled)
        self.start_button.setEnabled(self._mode_code() is not None)
        self.q_hint_label.setVisible(self.pages.currentIndex() == 0 and self._mode_code() is not None)

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

    def _handle_runtime_option_change(self, *_args):
        if self._suspend_updates:
            return
        self._save_config()
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
        elif sender is self.display_cb:
            key = "turn_on" if self.display_cb.isChecked() else "turn_off"
            self._speak_control_name(self._format_text(key, name=self._text("display_short")))
        elif sender is self.disable_accel_cb:
            key = "turn_on" if self.disable_accel_cb.isChecked() else "turn_off"
            self._speak_control_name(self._format_text(key, name=self._text("disable_accel_short")))

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
            display=self.display_cb.isChecked(),
            disable_accel=self.disable_accel_cb.isChecked(),
            enable_bubble_cursor=mode == "bubble",
            enable_semantic_pointing=mode == "semantic",
            enable_dynaspot=mode == "dynaspot",
            high_contrast_mode=self.high_contrast_cb.isChecked(),
            stronger_visual_cue=self._hidden_config.stronger_visual_cue,
            single_click_as_double_click=self._hidden_config.single_click_as_double_click,
            preset="TargetFinder" if mode == "targetfinder" else "Bubble Only" if mode == "bubble" else "Semantic Only" if mode == "semantic" else "DynaSpot" if mode == "dynaspot" else "",
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
        self.display_cb.setChecked(cfg.display)
        self.disable_accel_cb.setChecked(cfg.disable_accel)
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
        else:
            self._selected_mode = None

        self._selected_language = cfg.language if cfg.language in {"English", "French"} else "English"

        self._apply_language()
        self._apply_panel_style()
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
        cfg.enable_bubble_cursor = False
        cfg.enable_semantic_pointing = False
        cfg.enable_dynaspot = False
        cfg.display = False
        cfg.disable_accel = False
        cfg.high_contrast_mode = False
        cfg.enable_tts = False
        cfg.language = "English"
        cfg.preset = ""
        self._apply_config(cfg)

    # ---------------------------
    # Command / Process
    # ---------------------------
    def _build_command(self, cfg: PanelConfig):
        if cfg.preset == "TargetFinder":
            module_name = "target_finder_toolkit.targetfinder"
        elif cfg.enable_bubble_cursor:
            module_name = "target_finder_toolkit.bubblecursor"
        elif cfg.enable_dynaspot:
            module_name = "target_finder_toolkit.dynaspot"
        else:
            module_name = "target_finder_toolkit.semanticpointing"
        cmd = [sys.executable, "-m", module_name]
        cmd += ["--change-thresh", str(cfg.change_thresh)]
        cmd += ["--capture-interval", str(cfg.capture_interval)]
        cmd += ["--confidence", str(cfg.confidence)]
        cmd += ["--iou", str(cfg.iou)]
        if cfg.model_path:
            cmd += ["--model-path", cfg.model_path]

        if cfg.enable_semantic_pointing and cfg.display:
            cmd.append("--display")
        if cfg.enable_semantic_pointing and cfg.disable_accel:
            cmd.append("--disable-accel")
        return cmd

    def _is_demo_running(self):
        return self.process is not None and self.process.poll() is None

    def _launch_demo_for_config(self, cfg: PanelConfig, *, speak: bool):
        if cfg.model_path and not Path(cfg.model_path).is_file():
            self._set_status("invalid_model_path", speak=speak)
            return
        cmd = self._build_command(cfg)
        self.process = subprocess.Popen(cmd, cwd=str(self.project_root))
        if cfg.preset == "TargetFinder":
            self._set_status("running_targetfinder", speak=speak)
        elif cfg.enable_bubble_cursor:
            self._set_status("running_bubble", speak=speak)
        elif cfg.enable_dynaspot:
            self._set_status("running_dynaspot", speak=speak)
        else:
            self._set_status("running_semantic", speak=speak)

    def _stop_demo(self, silent: bool = False):
        if not self._is_demo_running():
            self.process = None
            if not silent:
                self._set_status("no_running", speak=False)
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        finally:
            self.process = None

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
