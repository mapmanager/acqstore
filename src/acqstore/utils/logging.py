"""Logging utilities for AcqStore.

Application entry points should call :func:`setup_logging` once at startup.
Library/module code should call :func:`get_logger`.

Example:
    from acqstore.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Loaded file")
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from platformdirs import user_config_dir

APP_NAME = "acqstore"
DEFAULT_LOG_FILENAME = "acqstore.log"
DISABLE_FILE_LOG_ENV = "ACQSTORE_DISABLE_FILE_LOG"

_LOG_FILE_PATH: Path | None = None
_HANDLER_MARKER = "_acqstore_handler"


def setup_logging(
    level: str | int = "INFO",
    *,
    app_name: str = APP_NAME,
    log_filename: str = DEFAULT_LOG_FILENAME,
    console: bool = True,
    file: bool = True,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> None:
    """Configure AcqStore package logging.

    This configures the ``acqstore`` logger rather than the Python root logger.
    Calling this function more than once replaces handlers previously installed
    by this function.

    Args:
        level: Console logging level. May be a string such as ``"INFO"`` or a
            standard logging integer such as ``logging.INFO``.
        app_name: Application name used by ``platformdirs`` for the log folder.
        log_filename: Name of the log file.
        console: If True, write logs to stderr.
        file: If True, write logs to a rotating file.
        max_bytes: Maximum size of the log file before rotation.
        backup_count: Number of rotated log files to keep.

    Returns:
        None.
    """
    resolved_level = _resolve_level(level)

    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    _remove_acqstore_handlers(logger)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(
            logging.Formatter(
                "[%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s"
            )
        )
        setattr(console_handler, _HANDLER_MARKER, True)
        logger.addHandler(console_handler)

    if file and not _file_log_disabled():
        global _LOG_FILE_PATH
        log_dir = Path(user_config_dir(app_name)) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        _LOG_FILE_PATH = log_dir / log_filename

        file_handler = logging.handlers.RotatingFileHandler(
            filename=_LOG_FILE_PATH,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] "
                "%(name)s:%(funcName)s:%(lineno)d: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        setattr(file_handler, _HANDLER_MARKER, True)
        logger.addHandler(file_handler)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return an AcqStore logger.

    Args:
        name: Logger name. Pass ``__name__`` from calling modules. If None,
            returns the top-level ``acqstore`` logger.

    Returns:
        A standard Python logger.
    """
    # return logging.getLogger(name or APP_NAME)
    if not name or name == '__main__':
        resolved_name = APP_NAME
    elif name == APP_NAME or name.startswith(f'{APP_NAME}.'):
        resolved_name = name
    else:
        resolved_name = f'{APP_NAME}.{name}'
    return logging.getLogger(resolved_name)

def get_log_file_path() -> Path | None:
    """Return the active log file path.

    Returns:
        Path to the AcqStore log file if file logging has been configured,
        otherwise None.
    """
    return _LOG_FILE_PATH


def _resolve_level(level: str | int) -> int:
    """Resolve a logging level from a string or integer.

    Args:
        level: Logging level as a string, such as ``"INFO"``, or integer.

    Returns:
        Integer logging level.

    Raises:
        ValueError: If a string level is not recognized.
        TypeError: If level is not str or int.
    """
    if isinstance(level, int):
        return level

    if not isinstance(level, str):
        raise TypeError(f"Invalid level type: {type(level)}")

    level_upper = level.upper()

    try:
        return logging._nameToLevel[level_upper]
    except KeyError:
        raise ValueError(f"Unknown logging level: {level!r}") from None


def _remove_acqstore_handlers(logger: logging.Logger) -> None:
    """Remove handlers previously installed by ``setup_logging``.

    Args:
        logger: Logger to clean.

    Returns:
        None.
    """
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()


def _file_log_disabled() -> bool:
    """Return whether file logging is disabled by environment variable.

    Returns:
        True if ``ACQSTORE_DISABLE_FILE_LOG`` is set to a truthy value.
    """
    value = os.getenv(DISABLE_FILE_LOG_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}
