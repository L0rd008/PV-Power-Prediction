"""Structured JSON logger shared across the service."""

import logging
import os
import sys
from datetime import date

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Daily rotating file handler (writes to ./logs/)
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"api_{date.today().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)

logger = logging.getLogger("pvlib-service")
