"""
logger.py

Utility for configuring logging in both console and file.
"""

import logging


def setup_logging(log_file: str = "pipeline.log"):
    """
    Setup logging to file and console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='a')
        ]
    )
