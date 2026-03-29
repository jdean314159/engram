"""Console logging setup.

Engram is often used as a library, so we avoid forcing global logging
configuration unless the host application hasn't configured logging.

If ENGRAM_LOG_LEVEL is set (e.g., INFO, DEBUG), we will configure a
StreamHandler to stderr at that level when no handlers are present.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_logging_if_needed(default_level: str = "INFO") -> None:
    """Configure console logging if the application hasn't configured it.

    - If the root logger already has handlers, we do nothing.
    - Otherwise, configure a basic StreamHandler to stderr.
    - Level is taken from ENGRAM_LOG_LEVEL if set, else default_level.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    level_name = (os.getenv("ENGRAM_LOG_LEVEL") or default_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stderr)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(level)
