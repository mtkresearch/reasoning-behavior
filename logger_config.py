"""
Logging configuration for the reasoning behavior project
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level=logging.INFO,
    console_level=logging.INFO,
    file_level=logging.DEBUG
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name (usually __name__)
        log_file: Optional path to log file. If None, only console logging is enabled
        level: Overall logger level (default: INFO)
        console_level: Console handler level (default: INFO)
        file_level: File handler level (default: DEBUG)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__, log_file='logs/experiment.log')
        >>> logger.info("Starting experiment")
        >>> logger.debug("Detailed debug information")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Console handler - show INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified) - log everything including DEBUG
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
