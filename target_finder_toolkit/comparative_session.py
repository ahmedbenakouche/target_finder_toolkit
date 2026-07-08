"""Run the healthy-participant comparative protocol.

The protocol runs both complete sessions:
1. the synthetic Fitts-with-distractors task;
2. the realistic annotated-screenshot task.

The order is counterbalanced by participant id: odd numbered participants start
with synthetic Fitts, even numbered participants start with the realistic task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from target_finder_toolkit.experimental_session import (
    add_session_technique_arguments,
    create_session_screen,
    is_english,
    task_runtime_args,
    write_event,
)
from target_finder_toolkit.synthetic_fitts_session import (
    DEFAULT_CONDITIONS_FILE,
    DEFAULT_SYNTHETIC_BLOCKS,
)
from target_finder_toolkit.experimental_task import DEFAULT_DATA_DIR, PROJECT_ROOT
from target_finder_toolkit.fitts_distractors_task import (
    DEFAULT_COUNTDOWN,
    DEFAULT_MAX_CLICKS,
    DEFAULT_TRIALS,
)


def _safe_id(participant_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in participant_id).strip("_")
    return safe or "participant"


def _participant_number(participant_id: str, seed: int | None = None) -> int:
    match = re.search(r"(\d+)\s*$", participant_id)
    if match is not None and seed is None:
        return max(1, int(match.group(1)))
    token = f"{participant_id}:{seed}" if seed is not None else participant_id
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], byteorder="big", signed=False) % 10_000) + 1


def comparative_task_order(participant_id: str, seed: int | None = None) -> list[str]:
    participant_number = _participant_number(participant_id, seed)
    if participant_number % 2 == 1:
        return ["synthetic_fitts", "realistic"]
    return ["realistic", "synthetic_fitts"]


def _base_session_args(args) -> list[str]:
    values = [
        "--language",
        args.language,
        "--participant",
        args.participant,
        "--trials-per-block",
        str(args.trials_per_block),
        "--countdown",
        str(args.countdown),
        "--max-clicks",
        str(args.max_clicks),
        "--cursor-log-hz",
        str(args.cursor_log_hz),
    ]
    if args.seed is not None:
        values += ["--seed", str(args.seed)]
    if args.windowed:
        values.append("--windowed")
    if args.no_technique_log:
        values.append("--no-technique-log")
    if args.no_preload_techniques:
        values.append("--no-preload-techniques")
    return values


def build_task_command(args, task_name: str, output_dir: Path) -> list[str]:
    if task_name == "synthetic_fitts":
        cmd = [
            sys.executable,
            "-m",
            "target_finder_toolkit.synthetic_fitts_session",
            *_base_session_args(args),
            "--synthetic-blocks",
            str(args.synthetic_blocks),
            "--conditions-file",
            str(args.conditions_file),
            "--output-dir",
            str(output_dir),
        ]
        if args.no_log:
            cmd.append("--no-log")
        return cmd + task_runtime_args(args)

    if task_name == "realistic":
        cmd = [
            sys.executable,
            "-m",
            "target_finder_toolkit.experimental_session",
            *_base_session_args(args),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(output_dir),
        ]
        if args.show_all_targets:
            cmd.append("--show-all-targets")
        if args.no_log:
            cmd.append("--no-log")
        return cmd + task_runtime_args(args)

    raise ValueError(f"Unknown comparative task: {task_name}")


def task_label(task_name: str, *, language: str) -> str:
    if is_english(language):
        return {
            "realistic": "the realistic screenshot experiment",
            "synthetic_fitts": "the synthetic Fitts-with-distractors experiment",
        }.get(task_name, task_name)
    return {
        "realistic": "l’expérience réaliste sur captures d’écran",
        "synthetic_fitts": "l’expérience Fitts synthétique avec distracteurs",
    }.get(task_name, task_name)


def task_description(task_name: str, *, language: str) -> str:
    if task_name == "realistic":
        if is_english(language):
            return (
                "In the next experiment, realistic interface screenshots will be displayed. "
                "On each trial, one target is highlighted with a red box. "
                "After the countdown, select the highlighted target using the current interaction technique."
            )
        return (
            "Dans la prochaine expérience, des captures d’écran d’interfaces réalistes seront affichées. "
            "À chaque essai, une cible sera encadrée en rouge. "
            "Après le compte à rebours, sélectionnez la cible indiquée avec la technique d’interaction en cours."
        )
    if is_english(language):
        return (
            "In the next experiment, a synthetic Fitts task with distractors will be displayed. "
            "Each trial starts from the blue point; move to the pink target and click it."
        )
    return (
        "Dans la prochaine expérience, une tâche synthétique de Fitts avec distracteurs sera affichée. "
        "Chaque essai commence depuis le point bleu ; déplacez-vous vers la cible rose et cliquez dessus."
    )


def show_between_task_pause(args, *, previous_task: str, next_task: str):
    from PyQt6 import QtWidgets

    screen = create_session_screen(windowed=args.windowed, language=args.language)
    previous_label = task_label(previous_task, language=args.language)
    next_label = task_label(next_task, language=args.language)
    if is_english(args.language):
        body = (
            f"The previous experiment is finished: {previous_label}.\n\n"
            f"The next experiment is: {next_label}."
        )
        hint = "When the break is over, click the button to start the next experiment."
        button_text = "Start next experiment"
    else:
        body = (
            f"L’expérience précédente est terminée : {previous_label}.\n\n"
            f"La prochaine expérience est : {next_label}."
        )
        hint = "Quand la pause est terminée, cliquez sur le bouton pour commencer l’expérience suivante."
        button_text = "Commencer l’expérience suivante"
    try:
        screen.show_content(
            title="Pause",
            body=body,
            hint=hint,
            button_text=button_text,
            pending_feedback=False,
        )
        aborted = screen.wait_for_continue()
        if aborted:
            screen.close()
            return True, None
        screen.close()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
        return False, None
    except Exception:
        screen.close()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
        raise


def show_completion_screen(args):
    from PyQt6 import QtWidgets

    screen = create_session_screen(windowed=args.windowed, language=args.language)
    if is_english(args.language):
        title = "Experiment completed"
        body = "Congratulations, you have completed this experimental session."
        hint = "Click Finish to close the experiment."
        button_text = "Finish"
    else:
        title = "Expérience terminée"
        body = "Félicitations, vous avez terminé cette session expérimentale."
        hint = "Cliquez sur Terminer pour fermer l’expérience."
        button_text = "Terminer"
    try:
        screen.show_content(
            title=title,
            body=body,
            hint=hint,
            button_text=button_text,
            pending_feedback=False,
        )
        screen.wait_for_continue()
    finally:
        screen.close()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run the healthy-participant comparative protocol: realistic task + synthetic Fitts task."
    )
    parser.add_argument("--participant", required=True, help="Participant id, e.g. P01")
    parser.add_argument("--language", choices=["French", "English"], default="French")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Annotated realistic screenshot dataset directory")
    parser.add_argument("--trials-per-block", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--synthetic-blocks", type=int, default=DEFAULT_SYNTHETIC_BLOCKS)
    parser.add_argument("--conditions-file", default=str(DEFAULT_CONDITIONS_FILE), help="CSV file containing ordered synthetic Fitts conditions")
    parser.add_argument("--countdown", type=float, default=DEFAULT_COUNTDOWN)
    parser.add_argument("--max-clicks", type=int, default=DEFAULT_MAX_CLICKS)
    parser.add_argument("--cursor-log-hz", type=float, default=30.0)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None, help="Optional base directory containing control_comparative, control_our_task, and control_fitts_synthetic")
    parser.add_argument("--show-all-targets", action="store_true")
    parser.add_argument("--no-technique-log", action="store_true")
    parser.add_argument("--no-log", action="store_true", help="Do not keep the comparative top-level log; passed to synthetic session.")
    parser.add_argument("--no-preload-techniques", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    add_session_technique_arguments(parser)
    return parser.parse_args(argv)


def run(args) -> int:
    order = comparative_task_order(args.participant, args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{_safe_id(args.participant)}_{stamp}_comparative"
    output_base = Path(args.output_dir).expanduser() if args.output_dir else PROJECT_ROOT
    comparative_root = output_base / "control_comparative" / session_id
    our_task_root = output_base / "control_our_task" / session_id
    synthetic_root = output_base / "control_fitts_synthetic" / session_id
    log_file = comparative_root / f"{session_id}_comparative.jsonl"

    if not args.no_log and not args.summary_only:
        comparative_root.mkdir(parents=True, exist_ok=True)
        our_task_root.mkdir(parents=True, exist_ok=True)
        synthetic_root.mkdir(parents=True, exist_ok=True)
        write_event(
            log_file,
            {
                "type": "comparative_session_start",
                "protocol": "complete_experiment_plus_synthetic_fitts",
                "protocol_label": "control_protocol_realistic_task_plus_synthetic_fitts",
                "participant_id": args.participant,
                "session_id": session_id,
                "task_order": order,
                "order_rule": "odd_participants_synthetic_first_even_participants_realistic_first",
                "trials_per_block": args.trials_per_block,
                "synthetic_blocks": args.synthetic_blocks,
                "conditions_file": str(args.conditions_file),
                "comparative_log_group": "control_comparative",
                "realistic_log_group": "control_our_task",
                "synthetic_log_group": "control_fitts_synthetic",
                "comparative_output_root": str(comparative_root),
                "realistic_output_root": str(our_task_root),
                "synthetic_output_root": str(synthetic_root),
            },
        )

    started = time.time()
    for task_index, task_name in enumerate(order, start=1):
        if task_name == "realistic":
            task_output_dir = our_task_root / f"{task_index:02d}_{task_name}"
            log_group = "control_our_task"
        else:
            task_output_dir = synthetic_root / f"{task_index:02d}_{task_name}"
            log_group = "control_fitts_synthetic"

        task_args = argparse.Namespace(**vars(args))
        cmd = build_task_command(task_args, task_name, task_output_dir)

        if args.summary_only:
            print(
                json.dumps(
                    {
                        "task": task_name,
                        "log_group": log_group,
                        "output_dir": str(task_output_dir),
                        "command": cmd,
                    },
                    ensure_ascii=False,
                )
            )
            continue

        if not args.no_log:
            write_event(
                log_file,
                {
                    "type": "comparative_task_start",
                    "task_index": task_index,
                    "task": task_name,
                    "task_label": task_label(task_name, language=args.language),
                    "log_group": log_group,
                    "output_dir": str(task_output_dir),
                    "command": cmd,
                    "ninja_auto_calibrate": bool(getattr(task_args, "ninja_auto_calibrate", False)),
                    "ninja_calibration_scope": "task_local",
                },
            )

        returncode = subprocess.call(cmd, cwd=str(PROJECT_ROOT))

        if not args.no_log:
            write_event(
                log_file,
                {
                    "type": "comparative_task_end",
                    "task_index": task_index,
                    "task": task_name,
                    "task_label": task_label(task_name, language=args.language),
                    "log_group": log_group,
                    "output_dir": str(task_output_dir),
                    "returncode": returncode,
                },
            )

        if returncode != 0:
            if not args.no_log:
                write_event(
                    log_file,
                    {
                        "type": "comparative_session_end",
                        "reason": "task_failed_or_aborted",
                        "failed_task": task_name,
                        "returncode": returncode,
                        "total_duration_sec": round(time.time() - started, 3),
                    },
                )
            return returncode

        if task_index < len(order):
            next_task = order[task_index]
            pause_started = time.time()
            if not args.no_log:
                write_event(
                    log_file,
                    {
                        "type": "comparative_pause_start",
                        "after_task_index": task_index,
                        "previous_task": task_name,
                        "next_task": next_task,
                    },
                )

            aborted, _pause_screen = show_between_task_pause(
                args,
                previous_task=task_name,
                next_task=next_task,
            )

            if not args.no_log:
                write_event(
                    log_file,
                    {
                        "type": "comparative_pause_end",
                        "after_task_index": task_index,
                        "previous_task": task_name,
                        "next_task": next_task,
                        "duration_sec": round(time.time() - pause_started, 3),
                        "aborted": bool(aborted),
                    },
                )

            if aborted:
                if not args.no_log:
                    write_event(
                        log_file,
                        {
                            "type": "comparative_session_end",
                            "reason": "keyboard_escape_on_between_task_pause",
                            "after_task": task_name,
                            "next_task": next_task,
                            "total_duration_sec": round(time.time() - started, 3),
                        },
                    )
                return 130

    if not args.summary_only:
        show_completion_screen(args)
    if not args.summary_only and not args.no_log:
        write_event(
            log_file,
            {
                "type": "comparative_session_end",
                "reason": "completed",
                "total_duration_sec": round(time.time() - started, 3),
            },
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
