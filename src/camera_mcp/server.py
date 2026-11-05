"""Server entry point. The MCP tools are added in milestone M3."""

import logging
import sys

from camera_mcp.config import ConfigError, load_config, resolve_config_path
from camera_mcp.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Validate the configuration; exit with status 1 if it is unusable."""
    setup_logging()
    try:
        config = load_config(resolve_config_path())
    except ConfigError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info("Configuration valid: %d camera(s).", len(config.cameras))


if __name__ == "__main__":
    main()
