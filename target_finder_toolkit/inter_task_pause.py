"""Standalone pause screen between two complete comparative tasks."""

from __future__ import annotations

import argparse
import sys

from target_finder_toolkit.experimental_session import create_session_screen, is_english


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Show a fullscreen pause between two complete tasks.")
    parser.add_argument("--language", choices=["French", "English"], default="French")
    parser.add_argument("--previous-label", required=True)
    parser.add_argument("--next-label", required=True)
    parser.add_argument("--windowed", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    screen = create_session_screen(windowed=args.windowed, language=args.language)
    if is_english(args.language):
        screen.show_content(
            title="Pause",
            body=(
                f"The previous experiment is finished: {args.previous_label}.\n\n"
                f"The next experiment is: {args.next_label}."
            ),
            hint="When the break is over, click the button to start the next experiment.",
            button_text="Start next experiment",
            pending_feedback=False,
        )
    else:
        screen.show_content(
            title="Pause",
            body=(
                f"L’expérience précédente est terminée : {args.previous_label}.\n\n"
                f"La prochaine expérience est : {args.next_label}."
            ),
            hint="Quand la pause est terminée, cliquez sur le bouton pour commencer l’expérience suivante.",
            button_text="Commencer l’expérience suivante",
            pending_feedback=False,
        )
    aborted = screen.wait_for_continue()
    screen.close()
    return 130 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
