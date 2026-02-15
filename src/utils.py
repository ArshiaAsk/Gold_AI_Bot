import logging
import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config

def setup_logger(name="GoldBot"):
    """
    Sets up a logger that writes to both file and console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create logs folder if not exists
    if not os.path.exists(config.paths.LOGS_DIR):
        os.makedirs(config.paths.LOGS_DIR)

    # Define Log File Name (e.g., bot_2025-12-21.log)
    log_filename = f"bot_{datetime.now().strftime('%Y-%m-%d')}.log"
    file_path = os.path.join(config.paths.LOGS_DIR, log_filename)

    # File Handler
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Only show INFO and above in console
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Create a global logger instance
logger = setup_logger()
