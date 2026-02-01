"""
Centralized logging configuration for PRKit (physical-reasoning-toolkit).

This module provides a unified logging interface that can be used across
all prkit_* packages for consistent logging behavior.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output with fallbacks."""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self._setup_colors()

    def _setup_colors(self):
        """Set up color support using ANSI codes only."""
        self._colors_available = False
        self.COLORS = {}

        # Use ANSI codes for colors
        try:
            import os
            import sys

            # Check if terminal supports colors
            if (
                hasattr(sys.stdout, "isatty")
                and sys.stdout.isatty()
                and os.getenv("TERM")
                and "color" in os.getenv("TERM", "")
            ):

                self.COLORS = {
                    "DEBUG": "\033[36m",  # Cyan
                    "INFO": "\033[32m",  # Green
                    "WARNING": "\033[33m",  # Yellow
                    "ERROR": "\033[31m",  # Red
                    "CRITICAL": "\033[35m",  # Magenta
                    "RESET": "\033[0m",  # Reset color
                }
                self._colors_available = True
            else:
                # Fallback with visual indicators when colors aren't supported
                self.COLORS = {
                    "DEBUG": "[DEBUG]",
                    "INFO": "[INFO]",
                    "WARNING": "[WARNING]",
                    "ERROR": "[ERROR]",
                    "CRITICAL": "[CRITICAL]",
                    "RESET": "",
                }
                self._colors_available = False
        except Exception:
            # Fallback with visual indicators on any error
            self.COLORS = {
                "DEBUG": "[DEBUG]",
                "INFO": "[INFO]",
                "WARNING": "[WARNING]",
                "ERROR": "[ERROR]",
                "CRITICAL": "[CRITICAL]",
                "RESET": "",
            }
            self._colors_available = False

    def format(self, record):
        """Format the log record with colors or visual indicators."""
        # Get the original formatted message
        formatted = super().format(record)

        # Add color/visual indicator for both console and file output
        level_name = record.levelname
        if level_name in self.COLORS:
            if self._colors_available:
                # Use colors (work in both console and file)
                formatted = formatted.replace(
                    level_name,
                    f"{self.COLORS[level_name]}{level_name}{self.COLORS['RESET']}",
                )
            else:
                # Use visual indicators (work in both console and file)
                formatted = formatted.replace(
                    level_name, f"{self.COLORS[level_name]} {level_name}"
                )

        return formatted


class PRKitLogger:
    """Centralized logger for PRKit (physical-reasoning-toolkit) packages with consistent configuration."""

    _loggers: Dict[str, logging.Logger] = {}
    _default_level = logging.INFO
    _default_format = "%(asctime)s - %(name)s - %(levelname)s [%(filename)s, %(lineno)d] - %(message)s"
    _default_date_format = "%Y-%m-%d %H:%M:%S"
    _colors_enabled = True  # Control whether colors are enabled

    class ConsoleFilter(logging.Filter):
        """Filter to mark console records for colored output."""

        def filter(self, record):
            record.console_output = True
            return True

    @classmethod
    def setup_global_config(
        cls,
        level: int = None,
        log_file: Optional[Path] = None,
        console_output: bool = True,
        format_string: str = None,
        date_format: str = None,
    ) -> None:
        """
        Set up global logging configuration for all PRKit packages.

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
        root_logger = logging.getLogger("prkit")
        root_logger.setLevel(cls._default_level)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(cls._default_level)

            # Use colored formatter for console output
            colored_formatter = ColoredFormatter(
                cls._default_format, cls._default_date_format
            )
            console_handler.setFormatter(colored_formatter)

            # Add a filter to mark console records
            console_handler.addFilter(cls.ConsoleFilter())
            root_logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(cls._default_level)

            # Use colored formatter for file output as well
            colored_formatter = ColoredFormatter(
                cls._default_format, cls._default_date_format
            )
            file_handler.setFormatter(colored_formatter)
            root_logger.addHandler(file_handler)

        # Set up environment-based configuration
        cls._setup_environment_config()

    @classmethod
    def _setup_environment_config(cls) -> None:
        """Set up logging configuration based on environment variables."""
        # Check for PRKIT_LOG_LEVEL environment variable
        env_level = os.getenv("PRKIT_LOG_LEVEL", "").upper()
        if env_level:
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            if env_level in level_map:
                cls._default_level = level_map[env_level]
                logging.getLogger("prkit").setLevel(cls._default_level)

        # Check for PRKIT_LOG_FILE environment variable
        env_log_file = os.getenv("PRKIT_LOG_FILE")
        if env_log_file:
            log_path = Path(env_log_file)
            # Add file handler directly instead of calling setup_global_config
            cls._add_file_handler_to_all(log_path)
        else:
            # Default log directory: {cwd}/prkit_logs/prkit.log
            default_log_dir = Path.cwd() / "prkit_logs"
            default_log_file = default_log_dir / "prkit.log"
            # Add file handler with default path
            cls._add_file_handler_to_all(default_log_file)

        # Check for PRKIT_LOG_CONSOLE environment variable
        env_console = os.getenv("PRKIT_LOG_CONSOLE", "true").lower()
        if env_console == "false":
            # Disable console output directly instead of calling setup_global_config
            cls._disable_console_output()

        # Check for PRKIT_LOG_COLORS environment variable
        env_colors = os.getenv("PRKIT_LOG_COLORS", "true").lower()
        if env_colors == "false":
            cls.disable_colors()
        elif cls.is_color_supported():
            cls.enable_colors()
        else:
            cls.disable_colors()

    @classmethod
    def get_logger(
        cls,
        name: str = None,
    ) -> logging.Logger:
        """
        Get a logger instance for the specified name.

        Args:
            name: Logger name (if None, uses the calling module's name)

        Returns:
            Logger instance with PRKit configuration
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
                    if "prkit" in module_name:
                        name = module_name
                    else:
                        name = "prkit"
                else:
                    name = "prkit"
            finally:
                try:
                    del frame
                except:
                    pass

        # Check if we already have this logger
        if name in cls._loggers:
            return cls._loggers[name]

        # Create new logger
        logger = logging.getLogger(name)

        # Set level if not already set
        if logger.level == logging.NOTSET:
            logger.setLevel(cls._default_level)

        # Add handlers from root logger if this logger has none
        if not logger.handlers:
            root_logger = logging.getLogger("prkit")
            for handler in root_logger.handlers:
                # Create a copy of the handler to avoid sharing
                if isinstance(handler, logging.FileHandler):
                    # Ensure parent directory exists before creating FileHandler
                    log_path = Path(handler.baseFilename)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    new_handler = logging.FileHandler(handler.baseFilename)
                    new_handler.setLevel(handler.level)

                    # Preserve colored formatting for file handlers
                    if isinstance(handler.formatter, ColoredFormatter):
                        new_handler.setFormatter(handler.formatter)
                    else:
                        # Use colored formatter for new file handlers
                        new_handler.setFormatter(
                            ColoredFormatter(
                                cls._default_format, cls._default_date_format
                            )
                        )

                    logger.addHandler(new_handler)
                elif isinstance(handler, logging.StreamHandler):
                    new_handler = logging.StreamHandler(handler.stream)
                    new_handler.setLevel(handler.level)

                    # Preserve colored formatting for console handlers
                    if isinstance(handler.formatter, ColoredFormatter):
                        new_handler.setFormatter(handler.formatter)
                        # Add console filter to mark records
                        new_handler.addFilter(cls.ConsoleFilter())
                    else:
                        new_handler.setFormatter(handler.formatter)

                    logger.addHandler(new_handler)
            
            # Disable propagation to avoid duplicate logs when handlers are added
            logger.propagate = False

        # Store logger
        cls._loggers[name] = logger

        return logger

    @classmethod
    def get_logger_with_selective_handlers(
        cls,
        name: str,
        log_file: Optional[Path] = None,
        console_output: bool = True,
        level: int = None,
    ) -> logging.Logger:
        """
        Get a logger instance with selective handler configuration.

        Args:
            name: Logger name
            log_file: Optional log file path (if None, no file handler)
            console_output: Whether to enable console output
            level: Logger level (if None, uses default)

        Returns:
            Logger instance with specified handler configuration
        """
        if level is None:
            level = cls._default_level

        # Check if we already have this logger
        if name in cls._loggers:
            logger = cls._loggers[name]
        else:
            # Create new logger
            logger = logging.getLogger(name)
            logger.setLevel(level)
            cls._loggers[name] = logger

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Add file handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # File gets all levels

            # Use colored formatter for file output as well
            colored_formatter = ColoredFormatter(
                cls._default_format, cls._default_date_format
            )
            file_handler.setFormatter(colored_formatter)
            logger.addHandler(file_handler)

        # Add console handler if enabled
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)  # Console respects specified level

            # Use colored formatter for console output
            colored_formatter = ColoredFormatter(
                cls._default_format, cls._default_date_format
            )
            console_handler.setFormatter(colored_formatter)

            # Add a filter to mark console records
            console_handler.addFilter(cls.ConsoleFilter())
            logger.addHandler(console_handler)

        return logger

    @classmethod
    def set_level(cls, level: int) -> None:
        """Set the global logging level for all PRKit loggers."""
        cls._default_level = level
        for logger in cls._loggers.values():
            logger.setLevel(level)

        # Also update root logger
        root_logger = logging.getLogger("prkit")
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)

    @classmethod
    def enable_colors(cls) -> None:
        """Enable colored output for console logging."""
        cls._colors_enabled = True
        cls._update_colors_for_all_handlers()

    @classmethod
    def disable_colors(cls) -> None:
        """Disable colored output for console logging."""
        cls._colors_enabled = False
        cls._update_colors_for_all_handlers()

    @classmethod
    def force_colors(cls) -> None:
        """Force enable colors even if terminal detection fails."""
        cls._colors_enabled = True
        # Force color detection to retry
        cls._update_colors_for_all_handlers()

    @classmethod
    def is_color_supported(cls) -> bool:
        """Check if the current terminal supports colors."""
        import os
        import sys

        # Check if we're in a terminal that supports colors
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return False

        # Check for common terminal types that support colors
        term = os.getenv("TERM", "")
        if term in ["xterm", "xterm-256color", "linux", "screen", "screen-256color"]:
            return True

        # Check for Windows color support (basic check)
        if os.name == "nt":
            # Windows terminals generally support colors
            return True

        return False

    @classmethod
    def _update_colors_for_all_handlers(cls) -> None:
        """Update color formatting for all existing handlers."""
        for logger in cls._loggers.values():
            for handler in logger.handlers:
                if (
                    isinstance(handler, logging.StreamHandler)
                    and handler.stream == sys.stdout
                ):
                    if cls._colors_enabled:
                        # Enable colors
                        if not isinstance(handler.formatter, ColoredFormatter):
                            handler.setFormatter(
                                ColoredFormatter(
                                    cls._default_format, cls._default_date_format
                                )
                            )
                            # Add console filter
                        handler.addFilter(cls.ConsoleFilter())
                    else:
                        # Disable colors
                        if isinstance(handler.formatter, ColoredFormatter):
                            handler.setFormatter(
                                logging.Formatter(
                                    cls._default_format, cls._default_date_format
                                )
                            )
                            # Remove console filters
                            handler.filters = [
                                f for f in handler.filters if not hasattr(f, "filter")
                            ]

        # Also update root logger
        root_logger = logging.getLogger("prkit")
        for handler in root_logger.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and handler.stream == sys.stdout
            ):
                if cls._colors_enabled:
                    # Enable colors
                    if not isinstance(handler.formatter, ColoredFormatter):
                        handler.setFormatter(
                            ColoredFormatter(
                                cls._default_format, cls._default_date_format
                            )
                        )
                        # Add console filter
                        handler.addFilter(cls.ConsoleFilter())
                else:
                    # Disable colors
                    if isinstance(handler.formatter, ColoredFormatter):
                        handler.setFormatter(
                            logging.Formatter(
                                cls._default_format, cls._default_date_format
                            )
                        )
                        # Remove console filters
                        handler.filters = [
                            f for f in handler.filters if not hasattr(f, "filter")
                        ]

    @classmethod
    def add_file_handler(cls, log_file: Path, level: int = None) -> None:
        """Add a file handler to all existing loggers."""
        if level is None:
            level = cls._default_level

        # Ensure parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use colored formatter for file output as well
        colored_formatter = ColoredFormatter(
            cls._default_format, cls._default_date_format
        )

        for logger_name, logger in cls._loggers.items():
            # Check if logger already has a file handler
            has_file_handler = any(
                isinstance(handler, logging.FileHandler) for handler in logger.handlers
            )

            if not has_file_handler:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level)
                file_handler.setFormatter(colored_formatter)
                logger.addHandler(file_handler)

        # Also add to root logger
        root_logger = logging.getLogger("prkit")
        has_file_handler = any(
            isinstance(handler, logging.FileHandler) for handler in root_logger.handlers
        )

        if not has_file_handler:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(colored_formatter)
            root_logger.addHandler(file_handler)

    @classmethod
    def _add_file_handler_to_all(cls, log_file: Path) -> None:
        """Add a file handler to all existing loggers without recursion."""
        if not log_file.parent.exists():
            log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use colored formatter for file output as well
        colored_formatter = ColoredFormatter(
            cls._default_format, cls._default_date_format
        )

        for logger_name, logger in cls._loggers.items():
            # Check if logger already has a file handler
            has_file_handler = any(
                isinstance(handler, logging.FileHandler) for handler in logger.handlers
            )

            if not has_file_handler:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(cls._default_level)
                file_handler.setFormatter(colored_formatter)
                logger.addHandler(file_handler)

        # Also add to root logger
        root_logger = logging.getLogger("prkit")
        has_file_handler = any(
            isinstance(handler, logging.FileHandler) for handler in root_logger.handlers
        )

        if not has_file_handler:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(cls._default_level)
            file_handler.setFormatter(colored_formatter)
            root_logger.addHandler(file_handler)

    @classmethod
    def _disable_console_output(cls) -> None:
        """Disable console output for all loggers without recursion."""
        for logger_name, logger in cls._loggers.items():
            # Remove console handlers
            console_handlers = [
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.StreamHandler)
                and handler.stream == sys.stdout
            ]
            for handler in console_handlers:
                logger.removeHandler(handler)

        # Also remove from root logger
        root_logger = logging.getLogger("prkit")
        console_handlers = [
            handler
            for handler in root_logger.handlers
            if isinstance(handler, logging.StreamHandler)
            and handler.stream == sys.stdout
        ]
        for handler in console_handlers:
            root_logger.removeHandler(handler)


# Initialize global configuration
PRKitLogger.setup_global_config()
