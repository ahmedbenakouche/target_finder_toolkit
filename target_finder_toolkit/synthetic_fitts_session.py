"""Run a full counterbalanced synthetic Fitts-with-distractors session."""

from __future__ import annotations

import argparse
import csv
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


DEFAULT_CONDITIONS_FILE = PROJECT_ROOT / "experiment_design" / "conditions.csv"
DEFAULT_SYNTHETIC_BLOCKS = 60
TECHNIQUE_LABEL_MAP = {
    "A": "mouse",
    "B": "semantic",
    "C": "bubble",
    "D": "dynaspot",
    "E": "ninja_cursors",
}
RHO_TO_DENSITY = {round(value, 6): name for name, value in DENSITY_VALUES.items()}


@dataclass(frozen=True)
class SyntheticSessionCondition:
    condition_index: int
    id_value: float
    difficulty: str
    density: str
    technique_label: str | None = None
    rho: float | None = None
    csv_row: int | None = None
    csv_order: int | None = None
    csv_token: str | None = None


@dataclass(frozen=True)
class SyntheticSessionBlock:
    block_id: str
    technique: str
    difficulty: str
    density: str
    id_value: float
    trials: int
    condition_index: int
    condition_count: int
    movement_count: int
    conditions: tuple[SyntheticSessionCondition, ...]
    technique_label: str | None = None
    rho: float | None = None
    csv_row: int | None = None
    csv_order: int | None = None
    csv_token: str | None = None


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


def _density_from_rho(rho: float) -> str:
    key = round(float(rho), 6)
    if key not in RHO_TO_DENSITY:
        raise ValueError(
            f"Unsupported rho={rho:g} in conditions file. Expected one of "
            f"{', '.join(str(value) for value in DENSITY_VALUES.values())}."
        )
    return RHO_TO_DENSITY[key]


def _parse_csv_condition_token(token: str, *, row_number: int, order: int) -> dict:
    parts = next(csv.reader([token], skipinitialspace=True))
    if len(parts) != 3:
        raise ValueError(
            f"Invalid condition token at row {row_number}, order {order}: {token!r}. "
            "Expected format: technique,ID,rho."
        )
    technique_label = parts[0].strip().upper()
    if technique_label not in TECHNIQUE_LABEL_MAP:
        raise ValueError(
            f"Unknown technique label {technique_label!r} at row {row_number}, order {order}. "
            f"Expected one of {', '.join(sorted(TECHNIQUE_LABEL_MAP))}."
        )
    id_value = float(parts[1].strip())
    rho = float(parts[2].strip())
    density = _density_from_rho(rho)
    return {
        "condition_index": order,
        "id_value": id_value,
        "difficulty": f"ID {id_value:g}",
        "density": density,
        "rho": rho,
        "technique_label": technique_label,
        "technique": TECHNIQUE_LABEL_MAP[technique_label],
        "csv_row": row_number,
        "csv_order": order,
        "csv_token": token.strip(),
    }


def _load_conditions_from_csv(
    *,
    participant_id: str,
    conditions_file: Path,
    seed: int | None = None,
) -> tuple[list[dict], dict]:
    path = Path(conditions_file).expanduser()
    participant_number = _participant_number(participant_id, seed=None)
    if not path.is_file():
        raise FileNotFoundError(path)

    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Conditions file is empty: {path}")
    row_index = participant_number - 1
    if row_index < 0 or row_index >= len(lines):
        raise ValueError(
            f"Participant {participant_id!r} maps to row {participant_number}, "
            f"but {path} only contains {len(lines)} participant rows."
        )

    raw_tokens = [token.strip() for token in lines[row_index].split(";") if token.strip()]
    conditions = [
        _parse_csv_condition_token(token, row_number=participant_number, order=index)
        for index, token in enumerate(raw_tokens, start=1)
    ]
    return conditions, {
        "method": "conditions_csv_participant_row_order",
        "condition_file": str(path),
        "participant_number": participant_number,
        "csv_row": participant_number,
        "available_conditions": len(conditions),
        "seed": seed,
    }


def _condition_to_dataclass(condition: dict) -> SyntheticSessionCondition:
    return SyntheticSessionCondition(
        condition_index=int(condition["condition_index"]),
        id_value=float(condition["id_value"]),
        difficulty=str(condition["difficulty"]),
        density=str(condition["density"]),
        technique_label=condition.get("technique_label"),
        rho=condition.get("rho", DENSITY_VALUES.get(str(condition["density"]))),
        csv_row=condition.get("csv_row"),
        csv_order=condition.get("csv_order"),
        csv_token=condition.get("csv_token"),
    )


def _group_conditions_by_consecutive_technique(conditions: list[dict]) -> list[list[dict]]:
    """The design CSV orders conditions in technique-level blocks.

    Each participant row contains 60 conditions. Consecutive equal technique
    labels form one interaction-technique block; inside it, the 12 ID x rho
    conditions are executed in the CSV order.
    """

    groups: list[list[dict]] = []
    current: list[dict] = []
    current_technique: str | None = None
    for condition in conditions:
        technique = str(condition["technique"])
        if current and technique != current_technique:
            groups.append(current)
            current = []
        current.append(condition)
        current_technique = technique
    if current:
        groups.append(current)
    return groups


def _make_block_from_condition_group(
    *,
    group_index: int,
    group: list[dict],
    trials_per_condition: int,
) -> SyntheticSessionBlock:
    if not group:
        raise ValueError("Cannot create a synthetic block from an empty condition group.")
    first = group[0]
    technique = str(first["technique"])
    technique_label = first.get("technique_label")
    block_id = f"{technique}_conditions_{group_index:02d}"
    conditions = tuple(_condition_to_dataclass(condition) for condition in group)
    return SyntheticSessionBlock(
        block_id=block_id,
        technique=technique,
        difficulty=str(first["difficulty"]),
        density=str(first["density"]),
        id_value=float(first["id_value"]),
        trials=trials_per_condition,
        condition_index=int(first["condition_index"]),
        condition_count=len(conditions),
        movement_count=len(conditions) * trials_per_condition,
        conditions=conditions,
        technique_label=technique_label,
        rho=first.get("rho", DENSITY_VALUES.get(str(first["density"]))),
        csv_row=first.get("csv_row"),
        csv_order=first.get("csv_order"),
        csv_token=first.get("csv_token"),
    )


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
    conditions_file: str | Path | None = DEFAULT_CONDITIONS_FILE,
) -> tuple[list[SyntheticSessionBlock], dict]:
    conditions_path = Path(conditions_file).expanduser() if conditions_file else None
    if conditions_path is not None and conditions_path.is_file():
        selected_conditions, assignment = _load_conditions_from_csv(
            participant_id=participant_id,
            conditions_file=conditions_path,
            seed=seed,
        )
        condition_count = max(1, min(int(block_count), len(selected_conditions)))
        selected_conditions = selected_conditions[:condition_count]
        order_indices = list(range(condition_count))
        plan_metadata = {
            "condition_assignment": {
                **assignment,
                "condition_count": condition_count,
                "selected_conditions": selected_conditions,
            },
            "technique_assignment": {
                "method": "conditions_csv_technique_labels",
                "mapping": TECHNIQUE_LABEL_MAP,
            },
            "block_ordering": {
                "method": "conditions_csv_order",
                "row_index": assignment["csv_row"] - 1,
                "order_indices": order_indices,
            },
        }
    else:
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

        for condition, technique in zip(selected_conditions, techniques):
            condition["technique"] = technique

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

        selected_conditions = [selected_conditions[index] for index in order_indices]
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

    condition_groups = _group_conditions_by_consecutive_technique(selected_conditions)
    base_blocks = [
        _make_block_from_condition_group(
            group_index=index,
            group=group,
            trials_per_condition=trials_per_block,
        )
        for index, group in enumerate(condition_groups, start=1)
    ]
    plan_metadata["condition_count"] = len(selected_conditions)
    plan_metadata["technique_block_count"] = len(base_blocks)
    plan_metadata["conditions_per_technique_block"] = [block.condition_count for block in base_blocks]
    plan_metadata["trials_per_condition"] = trials_per_block
    plan_metadata["total_movements"] = sum(block.movement_count for block in base_blocks)
    return base_blocks, plan_metadata


def make_synthetic_blocks(
    *,
    participant_id: str,
    block_count: int,
    trials_per_block: int,
    seed: int | None = None,
    conditions_file: str | Path | None = DEFAULT_CONDITIONS_FILE,
) -> list[SyntheticSessionBlock]:
    blocks, _metadata = make_synthetic_plan(
        participant_id=participant_id,
        block_count=block_count,
        trials_per_block=trials_per_block,
        seed=seed,
        conditions_file=conditions_file,
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
        "--condition-sequence-json",
        json.dumps([asdict(condition) for condition in block.conditions], separators=(",", ":")),
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
    parser.add_argument("--synthetic-blocks", type=int, default=DEFAULT_SYNTHETIC_BLOCKS, help="Number of synthetic conditions to run for this participant")
    parser.add_argument("--conditions-file", default=str(DEFAULT_CONDITIONS_FILE), help="CSV file containing one ordered condition row per participant")
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
        condition_sequence=[asdict(condition) for condition in block.conditions],
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
        conditions_file=args.conditions_file,
    )
    actual_id_values = sorted({float(condition.id_value) for block in ordered_blocks for condition in block.conditions})
    actual_densities = sorted({condition.density for block in ordered_blocks for condition in block.conditions})
    actual_rhos = sorted({
        float(condition.rho)
        for block in ordered_blocks
        for condition in block.conditions
        if condition.rho is not None
    })
    selected_conditions = plan_metadata.get("condition_assignment", {}).get("selected_conditions")
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
            "id_values": actual_id_values,
            "densities": actual_densities,
            "rho_values": actual_rhos,
            "conditions_file": str(Path(args.conditions_file).expanduser()) if args.conditions_file else None,
            "condition_pool": selected_conditions if selected_conditions is not None else _make_synthetic_condition_pool(),
            "condition_sampling": plan_metadata.get("condition_assignment", {}).get("method"),
            "plan_metadata": plan_metadata,
            "trials_per_condition": max(1, int(args.trials_per_block)),
            "trials_per_block": max(1, int(args.trials_per_block)),
            "block_count": len(ordered_blocks),
            "condition_count": sum(block.condition_count for block in ordered_blocks),
            "total_trials": sum(block.movement_count for block in ordered_blocks),
            "total_movements": sum(block.movement_count for block in ordered_blocks),
            "block_order": [asdict(block) for block in ordered_blocks],
            "counterbalancing": plan_metadata.get("block_ordering", {}).get("method"),
            "order_source": plan_metadata.get("block_ordering", {}).get("method"),
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
                    "technique_label": block.technique_label,
                    "difficulty": block.difficulty,
                    "id_value": block.id_value,
                    "density": block.density,
                    "rho": block.rho,
                    "condition_index": block.condition_index,
                    "condition_count": block.condition_count,
                    "movement_count": block.movement_count,
                    "trials_per_condition": block.trials,
                    "conditions": [asdict(condition) for condition in block.conditions],
                    "csv_row": block.csv_row,
                    "csv_order": block.csv_order,
                    "csv_token": block.csv_token,
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
                    "technique_label": block.technique_label,
                    "difficulty": block.difficulty,
                    "id_value": block.id_value,
                    "density": block.density,
                    "rho": block.rho,
                    "condition_index": block.condition_index,
                    "condition_count": block.condition_count,
                    "movement_count": block.movement_count,
                    "trials_per_condition": block.trials,
                    "conditions": [asdict(condition) for condition in block.conditions],
                    "csv_row": block.csv_row,
                    "csv_order": block.csv_order,
                    "csv_token": block.csv_token,
                    "returncode": returncode,
                    "elapsed_sec": round(elapsed, 3),
                },
            )
            if returncode != 0:
                reason = "keyboard_escape_in_block" if returncode == 130 else "block_failed"
                write_session_end(reason, failed_block_index=block_index, returncode=returncode)
                return returncode

            trial_offset += block.movement_count
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
            conditions_file=args.conditions_file,
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
