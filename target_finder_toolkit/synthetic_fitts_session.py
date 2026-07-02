"""Run a full counterbalanced synthetic Fitts-with-distractors session."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from target_finder_toolkit.experimental_session import (
    TECHNIQUES,
    PreloadedTechnique,
    add_session_technique_arguments,
    balanced_latin_square_indices,
    cleanup_session_resources,
    create_session_screen,
    show_break_screen,
    start_preloaded_techniques,
    wait_for_initial_ninja_calibration,
    wait_for_ninja_ready,
    wait_for_preloaded_startup,
    write_event,
    write_annotation_control_file,
    _technique_instruction,
    _participant_row_index,
    _drain_preloaded_outputs,
)
from target_finder_toolkit.fitts_distractors_task import (
    DEFAULT_COUNTDOWN,
    DEFAULT_CURSOR_HZ,
    DEFAULT_MAX_CLICKS,
    DEFAULT_TRIALS,
    DENSITY_VALUES,
    FittsDistractorsWindow,
    PROJECT_ROOT,
    SYNTHETIC_ID_VALUES,
    build_technique_command as build_fitts_technique_command,
)


@dataclass(frozen=True)
class SyntheticSessionBlock:
    block_id: str
    technique: str
    difficulty: str
    density: str
    id_value: float
    trials: int
    condition_index: int


def _safe_id(participant_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in participant_id).strip("_")
    return safe or "participant"


def _log_group_from_session_dir(session_dir: Path) -> str:
    try:
        return session_dir.resolve().relative_to(PROJECT_ROOT.resolve()).parts[0]
    except Exception:
        return session_dir.name


def _block_log_name(session_id: str, index: int, block: SyntheticSessionBlock) -> str:
    return f"{session_id}_block_{index:02d}_{block.block_id}.jsonl"


def _terminate_block_process(process: subprocess.Popen) -> int:
    if process.poll() is not None:
        return int(process.returncode or 0)
    try:
        process.terminate()
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
    return 130


def _block_log_has_event(path: Path, event_type: str) -> bool:
    if not path.exists():
        return False
    needle = f'"type": "{event_type}"'
    try:
        with path.open("r", encoding="utf-8") as fh:
            return any(needle in line for line in fh)
    except Exception:
        return False


def _stable_seed(participant_id: str, seed: int | None = None) -> int:
    token = f"{participant_id}:{seed}" if seed is not None else participant_id
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _participant_number(participant_id: str, seed: int | None = None) -> int:
    match = re.search(r"(\d+)\s*$", participant_id)
    if match is not None and seed is None:
        return max(1, int(match.group(1)))
    return (_stable_seed(participant_id, seed) % 10_000) + 1


def _make_synthetic_condition_pool() -> list[dict]:
    conditions = []
    index = 1
    for id_value in SYNTHETIC_ID_VALUES:
        for density in DENSITY_VALUES:
            conditions.append(
                {
                    "condition_index": index,
                    "id_value": float(id_value),
                    "difficulty": f"ID {float(id_value):g}",
                    "density": density,
                }
            )
            index += 1
    return conditions


def _select_condition_subset(
    *,
    participant_id: str,
    block_count: int,
    seed: int | None = None,
) -> tuple[list[dict], dict]:
    condition_pool = _make_synthetic_condition_pool()
    pool_size = len(condition_pool)
    block_count = max(1, min(int(block_count), pool_size))
    group_size = max(1, (pool_size + block_count - 1) // block_count)
    participant_number = _participant_number(participant_id, seed)
    group_index = (participant_number - 1) // group_size
    group_position = (participant_number - 1) % group_size

    group_token = f"synthetic_condition_group_{group_index}"
    rng = random.Random(_stable_seed(group_token, seed))
    shuffled = list(condition_pool)
    rng.shuffle(shuffled)

    start = (group_position * block_count) % pool_size
    selected = (shuffled + shuffled)[start : start + block_count]
    return selected, {
        "method": "participant_group_without_replacement_within_participant",
        "pool_size": pool_size,
        "block_count": block_count,
        "participant_number": participant_number,
        "group_size": group_size,
        "group_index": group_index,
        "group_position": group_position,
        "seed": seed,
    }


def make_synthetic_plan(
    *,
    participant_id: str,
    block_count: int,
    trials_per_block: int,
    seed: int | None = None,
) -> tuple[list[SyntheticSessionBlock], dict]:
    selected_conditions, assignment = _select_condition_subset(
        participant_id=participant_id,
        block_count=block_count,
        seed=seed,
    )
    block_count = len(selected_conditions)

    technique_cycle = list(TECHNIQUES)
    if technique_cycle:
        rotation = _participant_row_index(participant_id, seed, len(technique_cycle))
        technique_cycle = technique_cycle[rotation:] + technique_cycle[:rotation]

    techniques: list[str] = []
    while len(techniques) < block_count:
        techniques.extend(technique_cycle)
    techniques = techniques[:block_count]

    base_blocks: list[SyntheticSessionBlock] = []
    for index, (condition, technique) in enumerate(zip(selected_conditions, techniques), start=1):
        id_token = str(condition["id_value"]).replace(".", "_")
        density = condition["density"]
        block_id = f"{technique}_id{id_token}_{density}"
        base_blocks.append(
            SyntheticSessionBlock(
                block_id=block_id,
                technique=technique,
                difficulty=condition["difficulty"],
                density=density,
                id_value=condition["id_value"],
                trials=trials_per_block,
                condition_index=condition["condition_index"],
            )
        )

    if block_count > 1 and block_count % 2 == 0:
        square = balanced_latin_square_indices(block_count)
        row_index = _participant_row_index(participant_id, seed, len(square))
        order_indices = square[row_index]
        order_method = "balanced_latin_square"
    else:
        rng = random.Random(_stable_seed(f"{participant_id}:synthetic_order", seed))
        order_indices = list(range(block_count))
        rng.shuffle(order_indices)
        row_index = None
        order_method = "stable_random_order"

    blocks = [base_blocks[index] for index in order_indices]
    plan_metadata = {
        "condition_assignment": assignment,
        "technique_assignment": {
            "method": "participant_rotated_balanced_repetition_before_ordering",
            "techniques": list(TECHNIQUES),
            "participant_rotation": rotation if technique_cycle else 0,
            "cycle_for_this_participant": technique_cycle,
        },
        "block_ordering": {
            "method": order_method,
            "row_index": row_index,
            "order_indices": order_indices,
        },
    }
    return blocks, plan_metadata


def make_synthetic_blocks(
    *,
    participant_id: str,
    block_count: int,
    trials_per_block: int,
    seed: int | None = None,
) -> list[SyntheticSessionBlock]:
    blocks, _metadata = make_synthetic_plan(
        participant_id=participant_id,
        block_count=block_count,
        trials_per_block=trials_per_block,
        seed=seed,
    )
    return blocks


def _build_block_command(
    args,
    *,
    block: SyntheticSessionBlock,
    block_index: int,
    block_count: int,
    block_order: list[SyntheticSessionBlock],
    trial_offset: int,
    session_id: str,
    session_dir: Path,
    annotation_control_file: Path | None,
    ninja_control_file: Path | None,
    no_launch_technique: bool,
) -> list[str]:
    block_log_file = session_dir / _block_log_name(session_id, block_index, block)
    technique_log_file = session_dir / f"{block_log_file.stem}_{block.technique}_runtime.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "target_finder_toolkit.fitts_distractors_task",
        "--language",
        args.language,
        "--participant",
        args.participant,
        "--session-id",
        session_id,
        "--block-index",
        str(block_index),
        "--block-count",
        str(block_count),
        "--block-id",
        block.block_id,
        "--block-order",
        ",".join(item.block_id for item in block_order),
        "--trial-offset",
        str(trial_offset),
        "--technique",
        block.technique,
        "--trials",
        str(block.trials),
        "--difficulty",
        block.difficulty,
        "--density",
        block.density,
        "--id-value",
        str(block.id_value),
        "--countdown",
        str(args.countdown),
        "--max-clicks",
        str(args.max_clicks),
        "--log-file",
        str(block_log_file),
        "--technique-log-file",
        str(technique_log_file),
        "--cursor-log-hz",
        str(args.cursor_log_hz),
        "--technique-log-cursor-hz",
        str(args.technique_log_cursor_hz),
        "--change-thresh",
        str(args.change_thresh),
        "--capture-interval",
        str(args.capture_interval),
        "--confidence",
        str(args.confidence),
        "--iou",
        str(args.iou),
        "--filter",
        args.filter,
        "--filter-freq",
        str(args.filter_freq),
        "--filter-min-cutoff",
        str(args.filter_min_cutoff),
        "--filter-beta",
        str(args.filter_beta),
        "--filter-d-cutoff",
        str(args.filter_d_cutoff),
        "--dynaspot-min-speed",
        str(args.dynaspot_min_speed),
        "--dynaspot-spot-width",
        str(args.dynaspot_spot_width),
        "--dynaspot-lag",
        str(args.dynaspot_lag),
        "--dynaspot-reduce-time",
        str(args.dynaspot_reduce_time),
        "--ninja-camera-index",
        str(args.ninja_camera_index),
        "--ninja-spacing",
        str(args.ninja_spacing),
        "--ninja-gaze-smoothing",
        str(args.ninja_gaze_smoothing),
        "--ninja-gaze-gain-x",
        str(args.ninja_gaze_gain_x),
        "--ninja-gaze-gain-y",
        str(args.ninja_gaze_gain_y),
        "--ninja-gaze-offset-x",
        str(args.ninja_gaze_offset_x),
        "--ninja-gaze-offset-y",
        str(args.ninja_gaze_offset_y),
        "--ninja-selection-hold",
        str(args.ninja_selection_hold),
        "--ninja-calib-points",
        str(args.ninja_calib_points),
    ]
    if getattr(args, "ninja_hide_debug_status", False):
        cmd.append("--ninja-hide-debug-status")
    if no_launch_technique:
        cmd.append("--no-launch-technique")
        cmd.append("--keep-control-files")
    if annotation_control_file is not None:
        cmd += ["--annotation-control-file", str(annotation_control_file)]
    if ninja_control_file is not None and block.technique == "ninja_cursors":
        cmd += ["--ninja-control-file", str(ninja_control_file)]
    if args.windowed:
        cmd.append("--windowed")
    if args.no_technique_log:
        cmd.append("--no-technique-log")
    if args.no_log:
        cmd.append("--no-log")
    if args.semantic_display:
        cmd.append("--semantic-display")
    if args.semantic_disable_accel:
        cmd.append("--semantic-disable-accel")
    if args.ninja_screen_width_cm is not None:
        cmd += ["--ninja-screen-width-cm", str(args.ninja_screen_width_cm)]
    if args.ninja_screen_height_cm is not None:
        cmd += ["--ninja-screen-height-cm", str(args.ninja_screen_height_cm)]
    if args.ninja_lock_on_dwell:
        cmd.append("--ninja-lock-on-dwell")
    if args.ninja_hide_gaze_point:
        cmd.append("--ninja-hide-gaze-point")
    if getattr(args, "ninja_snap_system_cursor_to_active", False):
        cmd.append("--ninja-snap-system-cursor-to-active")
    if args.ninja_auto_calibrate:
        cmd.append("--ninja-auto-calibrate")
    if not args.ninja_without_targetfinder:
        cmd.append("--ninja-with-targetfinder")
    return cmd


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run a full synthetic Fitts-with-distractors session with counterbalanced blocks."
    )
    parser.add_argument("--participant", required=True, help="Participant id, e.g. P01")
    parser.add_argument("--language", choices=["French", "English"], default="French")
    parser.add_argument("--trials-per-block", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--synthetic-blocks", type=int, default=12, help="Number of sampled synthetic blocks per participant")
    parser.add_argument("--density", choices=tuple(DENSITY_VALUES), default="medium", help=argparse.SUPPRESS)
    parser.add_argument("--countdown", type=float, default=DEFAULT_COUNTDOWN)
    parser.add_argument("--max-clicks", type=int, default=DEFAULT_MAX_CLICKS)
    parser.add_argument("--cursor-log-hz", type=float, default=DEFAULT_CURSOR_HZ)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None, help="Directory for synthetic session and block logs")
    parser.add_argument("--no-technique-log", action="store_true")
    parser.add_argument("--no-log", action="store_true", help="Run without preserving synthetic session JSONL logs")
    parser.add_argument("--no-preload-techniques", action="store_true", help="Fallback: launch each technique inside each synthetic block")
    parser.add_argument("--summary-only", action="store_true")
    add_session_technique_arguments(parser)
    parser.set_defaults(model_path=None)
    return parser.parse_args(argv)


def run_block_in_process(
    args,
    block: SyntheticSessionBlock,
    *,
    session_id: str,
    block_index: int,
    block_count: int,
    block_order: list[SyntheticSessionBlock],
    trial_offset: int,
    session_dir: Path,
    block_log_file: Path,
    annotation_control_file: Path | None,
    ninja_control_file: Path | None,
    preloaded: PreloadedTechnique | None,
    transition_screen=None,
) -> int:
    """Run one synthetic Fitts block in this Qt process.

    Keeping the block window in the same process as the session transition
    screen avoids exposing the macOS desktop/menu bar between blocks.
    """

    from PyQt6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    technique_log_file = None if args.no_technique_log else (
        session_dir / f"{block_log_file.stem}_{block.technique}_runtime.jsonl"
    )
    task_annotation_control_file = annotation_control_file
    if task_annotation_control_file is None:
        task_annotation_control_file = block_log_file.with_name(f"block_{block_index:02d}_{block.block_id}.annotations.json")

    technique_command = None
    if preloaded is None and block.technique != "mouse":
        technique_args = argparse.Namespace(**vars(args))
        technique_args.technique = block.technique
        technique_command = build_fitts_technique_command(
            technique_args,
            technique_log_file,
            task_annotation_control_file,
        )

    window = FittsDistractorsWindow(
        participant_id=args.participant,
        technique=block.technique,
        difficulty=block.difficulty,
        density=block.density,
        id_value=block.id_value,
        trials=block.trials,
        countdown=args.countdown,
        max_clicks=args.max_clicks,
        log_file=block_log_file,
        cursor_log_hz=args.cursor_log_hz,
        technique_command=technique_command,
        technique_log_file=technique_log_file,
        annotation_control_file=task_annotation_control_file,
        language=args.language,
        seed=None,
        ninja_control_file=ninja_control_file if block.technique == "ninja_cursors" else None,
        external_technique_active=bool(preloaded),
        cleanup_control_files=not bool(preloaded),
        quit_application_on_complete=False,
        session_metadata={
            "session_id": session_id,
            "block_index": block_index,
            "block_count": block_count,
            "block_id": block.block_id,
            "block_order": ",".join(item.block_id for item in block_order),
            "trial_offset": trial_offset,
        },
    )
    block_loop = QtCore.QEventLoop()
    exit_holder = {"code": 0}

    def _finish_block(code: int):
        exit_holder["code"] = int(code)
        if block_loop.isRunning():
            block_loop.exit(int(code))

    window.finished.connect(_finish_block)
    if args.windowed:
        window.show()
    else:
        window.show_desktop_fullscreen()
    app.processEvents()
    if transition_screen is not None:
        QtCore.QTimer.singleShot(
            150,
            lambda: transition_screen.show_background_behind(clear_content=False),
        )
        app.processEvents()

    block_loop.exec()
    exit_code = int(window._exit_code or exit_holder["code"] or 0)
    window.close()
    window.deleteLater()
    app.processEvents()
    return exit_code


def run_session(args) -> int:
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{_safe_id(args.participant)}_{session_stamp}_synthetic_fitts"
    temp_dir_obj = None
    if args.no_log:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="target_finder_synthetic_session_")
        session_dir = Path(temp_dir_obj.name) / session_id
        args.no_technique_log = True
    else:
        session_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else PROJECT_ROOT / "control_fitts_synthetic" / session_id
        )
    session_dir.mkdir(parents=True, exist_ok=True)
    session_log_file = session_dir / f"{session_id}_session.jsonl"

    ordered_blocks, plan_metadata = make_synthetic_plan(
        participant_id=args.participant,
        block_count=max(1, int(args.synthetic_blocks)),
        trials_per_block=max(1, int(args.trials_per_block)),
        seed=args.seed,
    )
    total_started = time.monotonic()
    write_event(
        session_log_file,
        {
            "type": "session_start",
            "participant_id": args.participant,
            "session_id": session_id,
            "task": "synthetic_fitts_distractors_session",
            "task_label": "control_synthetic_fitts_with_distractors_task",
            "log_group": _log_group_from_session_dir(session_dir),
            "techniques": list(TECHNIQUES),
            "id_values": list(SYNTHETIC_ID_VALUES),
            "densities": list(DENSITY_VALUES.keys()),
            "condition_pool": _make_synthetic_condition_pool(),
            "condition_sampling": "participant_group_randomized_without_replacement",
            "plan_metadata": plan_metadata,
            "trials_per_block": max(1, int(args.trials_per_block)),
            "block_count": len(ordered_blocks),
            "total_trials": sum(block.trials for block in ordered_blocks),
            "block_order": [asdict(block) for block in ordered_blocks],
            "counterbalancing": "synthetic_blocks_ordered_with_balanced_latin_square_when_possible",
            "source": "synthetic_generated_targets_fake_targetfinder",
            "session_dir": str(session_dir),
        },
    )

    from PyQt6 import QtWidgets

    session_end_written = False

    def write_session_end(reason: str, **fields):
        nonlocal session_end_written
        if session_end_written:
            return
        session_end_written = True
        write_event(
            session_log_file,
            {
                "type": "session_end",
                "reason": reason,
                "total_duration_sec": round(time.monotonic() - total_started, 3),
                **fields,
            },
        )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    session_screen = create_session_screen(windowed=args.windowed, language=args.language)
    ninja_control_file = session_dir / "ninja_cursors.control"
    ninja_control_file.write_text("paused", encoding="utf-8")
    preloaded_processes: dict[str, PreloadedTechnique] = {}

    try:
        session_screen.show_content(
            title="Experiment initialization" if args.language == "English" else "Initialisation de l'expérience",
            body=(
                "Preparing the synthetic Fitts-with-distractors session.\n"
                "Loading the interaction techniques before the first block..."
                if args.language == "English"
                else "Préparation de la session synthétique de Fitts avec distracteurs.\n"
                "Chargement des techniques d'interaction avant le premier bloc..."
            ),
            hint="Please wait." if args.language == "English" else "Veuillez patienter.",
            button_text=None,
        )
        write_event(session_log_file, {"type": "initialization_start"})

        if not args.no_preload_techniques:
            preloaded_processes = start_preloaded_techniques(
                args,
                session_id=session_id,
                output_dir=session_dir,
                session_log=session_log_file,
                ninja_control_file=ninja_control_file,
            )
            wait_for_preloaded_startup(
                preloaded_processes,
                session_log_file,
                seconds=3.0,
                app=app,
                abort_check=lambda: bool(session_screen.aborted),
            )
            if session_screen.aborted:
                write_session_end("keyboard_escape_during_initialization")
                return 130

            if args.ninja_auto_calibrate:
                session_screen.show_content(
                    title="Experiment initialization" if args.language == "English" else "Initialisation de l'expérience",
                    body=(
                        "Preparing eye tracking.\n"
                        "Please wait while Ninja Cursors finishes initializing before calibration..."
                        if args.language == "English"
                        else "Préparation du suivi du regard.\n"
                        "Veuillez patienter pendant l'initialisation de Ninja Cursors avant la calibration..."
                    ),
                    hint=(
                        "This step may take a few seconds depending on the machine."
                        if args.language == "English"
                        else "Cette étape peut prendre quelques secondes selon la machine."
                    ),
                    button_text=None,
                )
                ninja_ready = wait_for_ninja_ready(
                    preloaded_processes,
                    session_log_file,
                    app=app,
                    abort_check=lambda: bool(session_screen.aborted),
                )
                if ninja_ready == "aborted":
                    write_session_end("keyboard_escape_during_initialization")
                    return 130
                if ninja_ready == "exited":
                    write_session_end("ninja_exited_during_initialization")
                    return 130
                if ninja_ready == "timeout":
                    write_session_end("ninja_ready_timeout")
                    return 1

                session_screen.show_content(
                    title="Eye-tracking calibration" if args.language == "English" else "Calibration du regard",
                    body=(
                        "Before the experiment starts, an eye-tracking calibration will be performed.\n"
                        "Look at each red point without moving your head until the next point appears."
                        if args.language == "English"
                        else "Avant de commencer l'expérience, une calibration du regard va être effectuée.\n"
                        "Regardez chaque point rouge sans bouger la tête jusqu'au point suivant."
                    ),
                    hint="Click Start when you are ready." if args.language == "English" else "Cliquez sur Commencer quand vous êtes prêt(e).",
                    button_text="Start calibration" if args.language == "English" else "Commencer la calibration",
                )
                write_event(session_log_file, {"type": "calibration_instructions"})
                if session_screen.wait_for_continue():
                    write_session_end("keyboard_escape_before_calibration")
                    return 130

                session_screen.show_content(
                    title="Calibration in progress" if args.language == "English" else "Calibration en cours",
                    body=(
                        "Look at the red point displayed on the screen until calibration is finished."
                        if args.language == "English"
                        else "Regardez le point rouge affiché à l'écran jusqu'à la fin de la calibration."
                    ),
                    hint=(
                        "Do not click and avoid moving your head during this step."
                        if args.language == "English"
                        else "Ne cliquez pas et évitez de bouger la tête pendant cette étape."
                    ),
                    button_text=None,
                    level_offset=0,
                )
                ninja_control_file.write_text("calibrate", encoding="utf-8")
                write_event(session_log_file, {"type": "calibration_start_requested"})
                calibration_result = wait_for_initial_ninja_calibration(
                    preloaded_processes,
                    session_log_file,
                    app=app,
                    abort_check=lambda: bool(session_screen.aborted),
                )
                ninja_control_file.write_text("paused", encoding="utf-8")
                if calibration_result in {"aborted", "cancelled"}:
                    write_session_end(
                        "keyboard_escape_during_calibration"
                        if calibration_result == "aborted"
                        else "calibration_cancelled"
                    )
                    return 130
                if calibration_result == "exited":
                    write_session_end("ninja_exited_during_calibration")
                    return 130
                if calibration_result == "timeout":
                    write_session_end("ninja_calibration_timeout")
                    return 1

        write_event(session_log_file, {"type": "initialization_end"})
        session_screen.show_content(
            title="Synthetic Fitts task" if args.language == "English" else "Tâche synthétique de Fitts",
            body=(
                "Each trial starts from the blue point.\n"
                "After the countdown, move to the pink target and click it.\n"
                "Grey circles are distractors. Use the current interaction technique.\n"
                "Press Esc to stop the session."
                if args.language == "English"
                else "Chaque essai commence depuis le point bleu.\n"
                "Après le compte à rebours, déplacez-vous vers la cible rose et cliquez dessus.\n"
                "Les cercles gris sont des distracteurs. Utilisez la technique d'interaction en cours.\n"
                "Appuyez sur Échap pour arrêter la session."
            ),
            hint="Click Start when you are ready." if args.language == "English" else "Cliquez sur Commencer quand vous êtes prêt(e).",
            button_text="Start experiment" if args.language == "English" else "Commencer l'expérience",
        )
        write_event(session_log_file, {"type": "experiment_start_instructions"})
        if session_screen.wait_for_continue():
            write_session_end("keyboard_escape_before_first_block")
            return 130

        if ordered_blocks and ordered_blocks[0].technique == "ninja_cursors":
            session_screen.show_content(
                title="Ninja Cursors",
                body=_technique_instruction(ordered_blocks[0], language=args.language),
                hint=(
                    "Click Continue when you are ready."
                    if args.language == "English"
                    else "Cliquez sur Continuer quand vous êtes prêt(e)."
                ),
                button_text="Continue" if args.language == "English" else "Continuer",
            )
            write_event(
                session_log_file,
                {
                    "type": "technique_instructions",
                    "technique": "ninja_cursors",
                    "before_block_index": 1,
                    "block_id": ordered_blocks[0].block_id,
                },
            )
            if session_screen.wait_for_continue():
                write_session_end("keyboard_escape_before_first_block")
                return 130

        trial_offset = 0
        for block_index, block in enumerate(ordered_blocks, start=1):
            app.processEvents()
            _drain_preloaded_outputs(preloaded_processes, session_log_file)
            preloaded = preloaded_processes.get(block.technique)
            if preloaded is not None and preloaded.process.poll() is not None:
                write_session_end(
                    "preloaded_technique_exited",
                    failed_block_index=block_index,
                    failed_block_id=block.block_id,
                    technique=block.technique,
                    exit_code=preloaded.process.poll(),
                )
                return preloaded.process.poll() or 1

            block_log_file = session_dir / _block_log_name(session_id, block_index, block)
            cmd = _build_block_command(
                args,
                block=block,
                block_index=block_index,
                block_count=len(ordered_blocks),
                block_order=ordered_blocks,
                trial_offset=trial_offset,
                session_id=session_id,
                session_dir=session_dir,
                annotation_control_file=preloaded.annotation_control_file if preloaded else None,
                ninja_control_file=ninja_control_file if preloaded_processes else None,
                no_launch_technique=bool(preloaded),
            )
            started = time.monotonic()
            write_event(
                session_log_file,
                {
                    "type": "block_start",
                    "block_index": block_index,
                    "block_count": len(ordered_blocks),
                    "trial_offset": trial_offset,
                    "block_id": block.block_id,
                    "technique": block.technique,
                    "difficulty": block.difficulty,
                    "id_value": block.id_value,
                    "density": block.density,
                    "condition_index": block.condition_index,
                    "trials": block.trials,
                    "command": cmd,
                    "block_log_file": str(block_log_file),
                    "preloaded_technique": bool(preloaded),
                    "annotation_control_file": str(preloaded.annotation_control_file) if preloaded else None,
                    "block_runtime": "in_process_fitts_distractors_task",
                },
            )
            returncode = run_block_in_process(
                args,
                block,
                session_id=session_id,
                block_index=block_index,
                block_count=len(ordered_blocks),
                block_order=ordered_blocks,
                trial_offset=trial_offset,
                session_dir=session_dir,
                block_log_file=block_log_file,
                annotation_control_file=preloaded.annotation_control_file if preloaded else None,
                ninja_control_file=ninja_control_file if preloaded_processes else None,
                preloaded=preloaded,
                transition_screen=session_screen,
            )
            elapsed = time.monotonic() - started
            _drain_preloaded_outputs(preloaded_processes, session_log_file)
            if preloaded:
                write_annotation_control_file(preloaded.annotation_control_file, state="inactive")
            ninja_control_file.write_text("paused", encoding="utf-8")
            write_event(
                session_log_file,
                {
                    "type": "block_end",
                    "block_index": block_index,
                    "block_count": len(ordered_blocks),
                    "trial_offset": trial_offset,
                    "block_id": block.block_id,
                    "technique": block.technique,
                    "difficulty": block.difficulty,
                    "id_value": block.id_value,
                    "density": block.density,
                    "condition_index": block.condition_index,
                    "returncode": returncode,
                    "elapsed_sec": round(elapsed, 3),
                },
            )
            if returncode != 0:
                reason = "keyboard_escape_in_block" if returncode == 130 else "block_failed"
                write_session_end(reason, failed_block_index=block_index, returncode=returncode)
                return returncode

            trial_offset += block.trials
            if block_index < len(ordered_blocks):
                next_block = ordered_blocks[block_index]
                pause_started = time.time()
                write_event(
                    session_log_file,
                    {
                        "type": "pause_start",
                        "after_block_index": block_index,
                        "current_block_id": block.block_id,
                        "next_block_id": next_block.block_id,
                        "current_block": asdict(block),
                        "next_block": asdict(next_block),
                    },
                )
                break_aborted = show_break_screen(
                    current_block=block,
                    next_block=next_block,
                    windowed=args.windowed,
                    language=args.language,
                    screen=session_screen,
                    current_block_index=block_index,
                    next_block_index=block_index + 1,
                    block_count=len(ordered_blocks),
                )
                pause_duration = time.time() - pause_started
                _drain_preloaded_outputs(preloaded_processes, session_log_file)
                write_event(
                    session_log_file,
                    {
                        "type": "pause_end",
                        "after_block_index": block_index,
                        "current_block_id": block.block_id,
                        "next_block_id": next_block.block_id,
                        "duration_sec": round(pause_duration, 3),
                        "aborted": bool(break_aborted),
                    },
                )
                if break_aborted:
                    write_session_end("keyboard_escape_on_break", after_block_index=block_index)
                    return 130

        write_session_end("completed")
        return 0
    finally:
        cleanup_session_resources(
            preloaded_processes=preloaded_processes,
            session_log=session_log_file,
            ninja_control_file=ninja_control_file,
            session_screen=session_screen,
            app=app,
        )
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summary_only:
        ordered_blocks, plan_metadata = make_synthetic_plan(
            participant_id=args.participant,
            block_count=max(1, int(args.synthetic_blocks)),
            trials_per_block=max(1, int(args.trials_per_block)),
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "plan_metadata": plan_metadata,
                    "blocks": [asdict(block) for block in ordered_blocks],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    return run_session(args)


if __name__ == "__main__":
    raise SystemExit(main())
