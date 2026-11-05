"""Logging configuration. Never logs to stdout: stdout is the MCP protocol channel."""

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Route all log output to stderr, replacing any previously installed root handlers."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.basicConfig(level=level, handlers=[handler], force=True)
