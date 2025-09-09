from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO, format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (default: includes timestamp, level, and message)
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger('paper2sw')
    logger.setLevel(level)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs by setting propagate to False
    logger.propagate = False
    
    return logger


def get_logger(name: str = 'paper2sw') -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (default: 'paper2sw')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)