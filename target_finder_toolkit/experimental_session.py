"""Run a full counterbalanced controlled experiment session."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from target_finder_toolkit.experimental_task import DEFAULT_DATA_DIR, PROJECT_ROOT


TECHNIQUES = ("bubble", "dynaspot", "semantic", "ninja_cursors")
DIFFICULTIES = ("easy", "medium", "hard")
TECHNIQUE_LABELS = {
    "bubble": "Bubble Cursor",
    "dynaspot": "DynaSpot",
    "semantic": "Pointage sémantique",
    "ninja_cursors": "Ninja Cursors",
}


@dataclass(frozen=True)
class ExperimentBlock:
    block_id: str
    technique: str
    difficulty: str
    trials: int


def make_blocks(trials_per_block: int) -> list[ExperimentBlock]:
    blocks: list[ExperimentBlock] = []
    for technique in TECHNIQUES:
        for difficulty in DIFFICULTIES:
            blocks.append(
                ExperimentBlock(
                    block_id=f"{technique}_{difficulty}",
                    technique=technique,
                    difficulty=difficulty,
                    trials=int(trials_per_block),
                )
            )
    return blocks


def counterbalanced_order(
    blocks: list[ExperimentBlock],
    *,
    participant_id: str,
    seed: int | None = None,
) -> list[ExperimentBlock]:
    """Return a deterministic participant-specific block order."""
    rng_seed = f"{participant_id}:{seed}" if seed is not None else participant_id
    rng = random.Random(rng_seed)
    ordered = list(blocks)
    rng.shuffle(ordered)
    return ordered


def write_event(log_file: Path, payload: dict):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp": time.time(), **payload}, ensure_ascii=False) + "\n")


def _format_block_label(block: ExperimentBlock) -> str:
    technique = TECHNIQUE_LABELS.get(block.technique, block.technique)
    return f"{technique} · difficulté {block.difficulty}"


def show_break_screen(
    *,
    current_block: ExperimentBlock,
    next_block: ExperimentBlock,
    seconds: float,
    windowed: bool,
    wait_for_click: bool,
):
    """Keep a fullscreen rest page visible between two external block processes."""
    if seconds <= 0 and not wait_for_click:
        return False

    from PyQt6 import QtCore, QtGui, QtWidgets

    try:
        from target_finder_toolkit.window_utils import raise_macos_window_above_system_ui
    except Exception:
        raise_macos_window_above_system_ui = None

    class BreakWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Pause")
            self.setCursor(QtCore.Qt.CursorShape.BlankCursor)
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            self.setStyleSheet(
                """
                QWidget {
                    background: #050505;
                    color: #f2f2f2;
                    font-family: Helvetica, Arial, sans-serif;
                }
                QLabel#Title {
                    font-size: 48px;
                    font-weight: 800;
                }
                QLabel#Body {
                    font-size: 28px;
                    font-weight: 500;
                    color: #d7d7d7;
                }
                QLabel#Countdown {
                    font-size: 72px;
                    font-weight: 900;
                    color: #b73832;
                }
                QLabel#Hint {
                    font-size: 22px;
                    color: #9a9a9a;
                }
                """
            )
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(64, 48, 64, 48)
            layout.setSpacing(18)
            layout.addStretch(1)

            title = QtWidgets.QLabel("Pause")
            title.setObjectName("Title")
            title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(title)

            body = QtWidgets.QLabel(
                f"Bloc terminé : {_format_block_label(current_block)}\n"
                f"Prochain bloc : {_format_block_label(next_block)}"
            )
            body.setObjectName("Body")
            body.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            body.setWordWrap(True)
            layout.addWidget(body)

            self.countdown_label = QtWidgets.QLabel("")
            self.countdown_label.setObjectName("Countdown")
            self.countdown_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.countdown_label)

            hint_text = (
                "Cliquez ou appuyez sur Entrée pour continuer."
                if wait_for_click
                else "Le prochain bloc va démarrer automatiquement."
            )
            hint = QtWidgets.QLabel(hint_text)
            hint.setObjectName("Hint")
            hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(hint)
            layout.addStretch(1)

            self.deadline = time.monotonic() + max(0.0, float(seconds))
            self.aborted = False
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.timer.start(100)
            self._escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self)
            self._escape_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            self._escape_shortcut.activated.connect(self._abort)
            self._q_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Q"), self)
            self._q_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            self._q_shortcut.activated.connect(self._abort)
            self._tick()

        def _tick(self):
            remaining = max(0.0, self.deadline - time.monotonic())
            if wait_for_click and remaining <= 0:
                self.countdown_label.setText("Prêt")
                return
            if remaining <= 0:
                self.close()
                return
            self.countdown_label.setText(f"{remaining:.0f} s")

        def keyPressEvent(self, event: QtGui.QKeyEvent):
            if event.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Q):
                self._abort()
                return
            if wait_for_click and self.deadline <= time.monotonic():
                if event.key() in (
                    QtCore.Qt.Key.Key_Return,
                    QtCore.Qt.Key.Key_Enter,
                    QtCore.Qt.Key.Key_Space,
                ):
                    self.close()
                    return
            super().keyPressEvent(event)

        def mousePressEvent(self, event: QtGui.QMouseEvent):
            if wait_for_click and self.deadline <= time.monotonic():
                self.close()
                return
            super().mousePressEvent(event)

        def _abort(self):
            self.aborted = True
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.exit(130)

        def closeEvent(self, event: QtGui.QCloseEvent):
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            if not self.aborted:
                QtWidgets.QApplication.instance().quit()
            super().closeEvent(event)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = BreakWindow()
    if windowed:
        window.resize(900, 520)
        window.show()
    else:
        window.showFullScreen()
    if raise_macos_window_above_system_ui is not None:
        raise_macos_window_above_system_ui(window)
    return app.exec() == 130 or bool(getattr(window, "aborted", False))


def build_block_command(
    args,
    block: ExperimentBlock,
    *,
    session_id: str,
    block_index: int,
    block_count: int,
    block_order: list[ExperimentBlock],
    trial_offset: int,
    block_log_file: Path,
    technique_log_file: Path,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "target_finder_toolkit.experimental_task",
        "--data-dir",
        str(args.data_dir),
        "--technique",
        block.technique,
        "--difficulty",
        block.difficulty,
        "--trials",
        str(block.trials),
        "--countdown",
        str(args.countdown),
        "--max-clicks",
        str(args.max_clicks),
        "--participant-id",
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
        "--log-file",
        str(block_log_file),
        "--technique-log-file",
        str(technique_log_file),
    ]
    if args.show_all_targets:
        cmd.append("--show-all-targets")
    if args.windowed:
        cmd.append("--windowed")
    if args.no_technique_log:
        cmd.append("--no-technique-log")
    if args.technique_start_delay is not None:
        cmd += ["--technique-start-delay", str(args.technique_start_delay)]
    return cmd + extra_args


def main():
    parser = argparse.ArgumentParser(
        description="Run a counterbalanced experimental session made of technique/difficulty blocks."
    )
    parser.add_argument("--participant", required=True, help="Participant id, e.g. P01")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Annotated dataset directory")
    parser.add_argument("--trials-per-block", type=int, default=20, help="Trials per technique/difficulty block")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds passed to each block")
    parser.add_argument("--max-clicks", type=int, default=1, help="Maximum clicks per trial")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed combined with participant id")
    parser.add_argument("--output-dir", default=None, help="Directory for session and block logs")
    parser.add_argument("--break-seconds", type=float, default=10.0, help="Rest time between blocks; 0 disables timed breaks")
    parser.add_argument("--pause-between-blocks", action="store_true", help="Wait for Enter between blocks")
    parser.add_argument("--show-all-targets", action="store_true", help="Debug: show all annotated targets")
    parser.add_argument("--windowed", action="store_true", help="Run each block in a window")
    parser.add_argument("--no-technique-log", action="store_true", help="Disable separate runtime technique logs")
    parser.add_argument("--technique-start-delay", type=float, default=None, help="Override technique startup delay")
    parser.add_argument("--dry-run", action="store_true", help="Print generated order without running blocks")
    args, extra_args = parser.parse_known_args()

    if args.trials_per_block <= 0:
        raise SystemExit("--trials-per-block must be positive")

    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{args.participant}_{session_stamp}"
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else PROJECT_ROOT / "logs" / "experimental_sessions" / session_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = make_blocks(args.trials_per_block)
    ordered_blocks = counterbalanced_order(
        blocks,
        participant_id=args.participant,
        seed=args.seed,
    )
    block_count = len(ordered_blocks)
    total_trials = sum(block.trials for block in ordered_blocks)
    session_log = output_dir / f"{session_id}_session.jsonl"

    print("experimental session")
    print(f"  participant={args.participant}")
    print(f"  session_id={session_id}")
    print(f"  blocks={block_count}")
    print(f"  total_trials={total_trials}")
    print("  order:")
    for index, block in enumerate(ordered_blocks, start=1):
        print(f"    {index:02d}. {block.block_id} ({block.trials} trials)")

    write_event(
        session_log,
        {
            "type": "session_start",
            "participant_id": args.participant,
            "session_id": session_id,
            "data_dir": str(args.data_dir),
            "trials_per_block": args.trials_per_block,
            "block_count": block_count,
            "total_trials": total_trials,
            "block_order": [asdict(block) for block in ordered_blocks],
            "extra_args": extra_args,
        },
    )

    if args.dry_run:
        write_event(session_log, {"type": "session_end", "reason": "dry_run"})
        return

    trial_offset = 0
    for block_index, block in enumerate(ordered_blocks, start=1):
        block_prefix = f"block_{block_index:02d}_{block.block_id}"
        block_log_file = output_dir / f"{block_prefix}_trials.jsonl"
        technique_log_file = output_dir / f"{block_prefix}_technique.jsonl"
        cmd = build_block_command(
            args,
            block,
            session_id=session_id,
            block_index=block_index,
            block_count=block_count,
            block_order=ordered_blocks,
            trial_offset=trial_offset,
            block_log_file=block_log_file,
            technique_log_file=technique_log_file,
            extra_args=extra_args,
        )

        write_event(
            session_log,
            {
                "type": "block_start",
                "block_index": block_index,
                "block_count": block_count,
                "trial_offset": trial_offset,
                **asdict(block),
                "block_log_file": str(block_log_file),
                "technique_log_file": str(technique_log_file),
                "command": cmd,
            },
        )
        print(f"\nstarting block {block_index}/{block_count}: {block.block_id}")
        started = time.time()
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        elapsed = time.time() - started
        write_event(
            session_log,
            {
                "type": "block_end",
                "block_index": block_index,
                "block_count": block_count,
                "trial_offset": trial_offset,
                **asdict(block),
                "returncode": result.returncode,
                "elapsed_sec": round(elapsed, 3),
            },
        )
        if result.returncode != 0:
            reason = "escape_in_block" if result.returncode == 130 else "block_failed"
            write_event(
                session_log,
                {
                    "type": "session_end",
                    "reason": reason,
                    "failed_block_index": block_index,
                    "returncode": result.returncode,
                },
            )
            raise SystemExit(result.returncode)

        trial_offset += block.trials
        if block_index < block_count:
            next_block = ordered_blocks[block_index]
            if args.pause_between_blocks or args.break_seconds > 0:
                print(
                    f"pause avant le prochain bloc: {next_block.block_id} "
                    f"({args.break_seconds:g}s)"
                )
                break_aborted = show_break_screen(
                    current_block=block,
                    next_block=next_block,
                    seconds=max(0.0, float(args.break_seconds)),
                    windowed=args.windowed,
                    wait_for_click=args.pause_between_blocks,
                )
                if break_aborted:
                    write_event(
                        session_log,
                        {
                            "type": "session_end",
                            "reason": "escape_on_break",
                            "after_block_index": block_index,
                        },
                    )
                    raise SystemExit(130)

    write_event(session_log, {"type": "session_end", "reason": "completed"})
    print(f"\nsession completed: {session_log}")


if __name__ == "__main__":
    main()
