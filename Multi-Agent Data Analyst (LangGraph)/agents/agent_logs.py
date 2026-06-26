import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import os

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
handler = TimedRotatingFileHandler(
    "logs/agent.log",
    encoding='utf-8',
    when="midnight",
    backupCount=30,
)

handler.setFormatter(logging.Formatter("%(asctime)s:\n%(message)s\n"))
logger.addHandler(handler)