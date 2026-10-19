#!/usr/bin/env python3
"""Choose and launch the local assistant interface."""

import argparse
import sys
from typing import Callable, Optional, Sequence


def choose_mode(input_fn: Callable[[str], str] = input) -> Optional[str]:
    print("Yerel Asistan")
    print("1) CLI sohbet arayüzü")
    print("2) Masaüstü uygulaması")
    print("q) Çıkış")
    while True:
        choice = input_fn("Seçimin (1/2/q): ").strip().lower()
        if choice in {"1", "cli"}:
            return "cli"
        if choice in {"2", "app"}:
            return "app"
        if choice in {"q", "quit", "exit"}:
            return None
        print("Geçersiz seçim. 1, 2 veya q gir.")


def launch(mode: str) -> None:
    if mode == "cli":
        from tui_app import main as run_cli

        run_cli()
        return
    if mode == "app":
        from desktop_app import main as run_desktop

        run_desktop()
        return
    raise ValueError(f"Bilinmeyen arayüz: {mode}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Yerel Ollama Asistanı")
    parser.add_argument(
        "--mode",
        choices=("cli", "app"),
        help="Seçim ekranını atla: cli veya app",
    )
    args = parser.parse_args(argv)
    mode = args.mode
    if mode is None:
        try:
            mode = choose_mode()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
    if mode is not None:
        launch(mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
