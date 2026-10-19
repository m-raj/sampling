"""Logging and progress-bar setup shared by the command-line scripts.

Scripts call ``setup_logging`` once at start-up; everything else only uses
``logging.getLogger(__name__)``.

Progress bars are shown only in an interactive terminal or a notebook. In a
batch job (e.g. SLURM, where output is redirected to a file) tqdm would fill
that file with bar redraws, so the bars are turned off and progress is logged
as plain lines instead.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

FORMAT = "%(asctime)s %(levelname)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)


def progress_bars_enabled() -> bool:
    """True in an interactive terminal or a Jupyter notebook, False in batch jobs."""
    return "ipykernel" in sys.modules or sys.stderr.isatty()


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Log to stdout and, if given, append to ``log_file``.

    Also sends warnings and uncaught exceptions (with traceback) to the log.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level, format=FORMAT, datefmt=DATE_FORMAT, handlers=handlers, force=True
    )
    # Lightning attaches its own console handler when it is imported before
    # this runs; route its messages through ours so they reach the file too.
    for name in ("lightning", "lightning.pytorch", "lightning.fabric"):
        library_logger = logging.getLogger(name)
        library_logger.handlers.clear()
        library_logger.propagate = True
    # matplotlib logs a font notice at INFO for every figure.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.captureWarnings(True)
    sys.excepthook = _log_uncaught_exception


def _log_uncaught_exception(exc_type, exc_value, traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, traceback)
        return
    logging.getLogger().critical(
        "Uncaught exception", exc_info=(exc_type, exc_value, traceback)
    )


def progress(
    iterable: Iterable[T],
    desc: str,
    total: int | None = None,
    unit: str = "it",
    log_fraction: float = 0.1,
) -> Iterator[T]:
    """Iterate with a tqdm bar, or, in batch jobs, log every ``log_fraction`` of the way."""
    if progress_bars_enabled():
        yield from tqdm(iterable, desc=desc, total=total, unit=unit, dynamic_ncols=True)
        return
    total = len(iterable) if total is None else total
    step = max(1, round(total * log_fraction))
    start = time.monotonic()
    for count, item in enumerate(iterable, start=1):
        yield item
        if count % step == 0 or count == total:
            logger.info(
                "%s: %d/%d %s (%.0f s)", desc, count, total, unit, time.monotonic() - start
            )
