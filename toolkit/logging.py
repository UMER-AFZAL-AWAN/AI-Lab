"""
toolkit.logging — structured, levelled logger used by all toolkit libraries.

Usage:
    from toolkit.logging import get_logger
    log = get_logger(__name__)
    log.info("RIFE", "normalising VFR → CFR 30 fps")
    log.step(2, 7, "Generating AnimateDiff loop")
    log.done("enhance_video", output_path)
"""

import logging
import sys
from pathlib import Path


# ── ANSI colours (disabled automatically on non-TTY) ────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREY   = "\033[90m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"

_USE_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if _USE_COLOUR else text


class ToolkitLogger:
    """
    Thin wrapper around stdlib logging that adds structured helpers:
      log.step(n, total, msg)   — "[3/7] msg"
      log.done(tag, path)       — green "✓ tag → path"
      log.info(tag, msg)        — "[tag] msg"
      log.warn(tag, msg)        — yellow warning
      log.error(tag, msg)       — red error (also goes to stderr)
    """

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def step(self, n: int, total: int, msg: str) -> None:
        label = _c(_CYAN, f"[{n}/{total}]")
        self._log.info("%s %s", label, msg)

    def done(self, tag: str, path: Path | str | None = None) -> None:
        tick = _c(_GREEN, "✓")
        tag_str = _c(_BOLD, tag)
        suffix = f" → {path}" if path else ""
        self._log.info("%s %s%s", tick, tag_str, suffix)

    def info(self, tag: str, msg: str) -> None:
        self._log.info("%s %s", _c(_GREY, f"[{tag}]"), msg)

    def warn(self, tag: str, msg: str) -> None:
        self._log.warning("%s %s", _c(_YELLOW, f"[{tag}]"), msg)

    def error(self, tag: str, msg: str) -> None:
        self._log.error("%s %s", _c(_RED, f"[{tag}]"), msg)


def get_logger(name: str) -> ToolkitLogger:
    """Return a ToolkitLogger for the given module name."""
    return ToolkitLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Call once at application entry-point (cli.py) to set up the root handler.
    Libraries never call this — only the app entry point does.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
