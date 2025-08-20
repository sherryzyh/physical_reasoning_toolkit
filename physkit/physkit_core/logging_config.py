"""
Centralized logging configuration for PhysKit packages.

This module provides a unified logging interface that can be used across
all physkit_* packages for consistent logging behavior.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import os


class PhysKitLogger:
    """Centralized logger for PhysKit packages with consistent configuration."""
    
    _loggers: Dict[str, logging.Logger] = {}
    _default_level = logging.INFO
    _default_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    _default_date_format = '%Y-%m-%d %H:%M:%S'
    
    @classmethod
    def setup_global_config(
        cls,
        level: int = None,
        log_file: Optional[Path] = None,
        console_output: bool = True,
        format_string: str = None,
        date_format: str = None
    ) -> None:
        """
        Set up global logging configuration for all PhysKit packages.
        
        Args:
            level: Global logging level (default: INFO)
            log_file: Optional global log file path
            console_output: Whether to output to console (default: True)
            format_string: Custom log format string
            date_format: Custom date format string
        """
        if level is not None:
            cls._default_level = level
        
        if format_string is not None:
            cls._default_format = format_string
            
        if date_format is not None:
            cls._default_date_format = date_format
        
        # Set up root logger
        root_logger = logging.getLogger("physkit")
        root_logger.setLevel(cls._default_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(cls._default_level)
            formatter = logging.Formatter(cls._default_format, cls._default_date_format)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(cls._default_level)
            formatter = logging.Formatter(cls._default_format, cls._default_date_format)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Set up environment-based configuration
        cls._setup_environment_config()
    
    @classmethod
    def _setup_environment_config(cls) -> None:
        """Set up logging configuration based on environment variables."""
        # Check for PHYSKIT_LOG_LEVEL environment variable
        env_level = os.getenv("PHYSKIT_LOG_LEVEL", "").upper()
        if env_level:
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL
            }
            if env_level in level_map:
                cls._default_level = level_map[env_level]
                logging.getLogger("physkit").setLevel(cls._default_level)
        
        # Check for PHYSKIT_LOG_FILE environment variable
        env_log_file = os.getenv("PHYSKIT_LOG_FILE")
        if env_log_file:
            log_path = Path(env_log_file)
            cls.setup_global_config(log_file=log_path)
        
        # Check for PHYSKIT_LOG_CONSOLE environment variable
        env_console = os.getenv("PHYSKIT_LOG_CONSOLE", "true").lower()
        if env_console == "false":
            cls.setup_global_config(console_output=False)
    
    @classmethod
    def get_logger(cls, name: str = None) -> logging.Logger:
        """
        Get a logger instance for the specified name.
        
        Args:
            name: Logger name (if None, uses the calling module's name)
            
        Returns:
            Logger instance with PhysKit configuration
        """
        if name is None:
            # Get the calling module's name
            import inspect
            frame = inspect.currentframe()
            try:
                # Go up one frame to get the caller
                caller_frame = frame.f_back
                if caller_frame:
                    module_name = inspect.getmodule(caller_frame).__name__
                    # Extract the relevant part for logging
                    if 'physkit' in module_name:
                        name = module_name
                    else:
                        name = "physkit"
                else:
                    name = "physkit"
            finally:
                del frame
        
        # Check if we already have this logger
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create new logger
        logger = logging.getLogger(name)
        
        # Set level if not already set
        if logger.level == logging.NOTSET:
            logger.setLevel(cls._default_level)
        
        # Store logger
        cls._loggers[name] = logger
        
        return logger
    
    @classmethod
    def set_level(cls, level: int) -> None:
        """Set the global logging level for all PhysKit loggers."""
        cls._default_level = level
        for logger in cls._loggers.values():
            logger.setLevel(level)
        
        # Also update root logger
        root_logger = logging.getLogger("physkit")
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
    
    @classmethod
    def add_file_handler(cls, log_file: Path, level: int = None) -> None:
        """Add a file handler to all existing loggers."""
        if level is None:
            level = cls._default_level
        
        formatter = logging.Formatter(cls._default_format, cls._default_date_format)
        
        for logger_name, logger in cls._loggers.items():
            # Check if logger already has a file handler
            has_file_handler = any(
                isinstance(handler, logging.FileHandler) 
                for handler in logger.handlers
            )
            
            if not has_file_handler:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        # Also add to root logger
        root_logger = logging.getLogger("physkit")
        has_file_handler = any(
            isinstance(handler, logging.FileHandler) 
            for handler in root_logger.handlers
        )
        
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)





# Initialize global configuration
PhysKitLogger.setup_global_config()
