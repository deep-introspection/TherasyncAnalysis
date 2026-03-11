"""
Logging setup for Therasync Pipeline.

This module provides centralized logging configuration with file rotation,
colored console output, and different log levels for different components.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import colorlog

    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


class LoggerSetup:
    """
    Centralized logging setup for the Therasync pipeline.

    Provides file logging with rotation, console logging with colors,
    and module-specific loggers.
    """

    def __init__(self, log_dir: str = "log", log_level: str = "INFO"):
        """
        Initialize logger setup.

        Args:
            log_dir: Directory for log files.
            log_level: Default logging level.
        """
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir.mkdir(exist_ok=True)

        # Store created loggers to avoid duplicates
        self._configured_loggers: Dict[str, logging.Logger] = {}

    def setup_root_logger(
        self,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        max_file_size: str = "10MB",
        backup_count: int = 5,
    ) -> logging.Logger:
        """
        Setup the root logger with console and file handlers.

        Args:
            console_level: Logging level for console output.
            file_level: Logging level for file output.
            max_file_size: Maximum size for each log file.
            backup_count: Number of backup files to keep.

        Returns:
            Configured root logger.
        """
        # Parse file size
        size_bytes = self._parse_file_size(max_file_size)

        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture all levels

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Console handler with colors (if available)
        console_handler = self._create_console_handler(console_level)
        root_logger.addHandler(console_handler)

        # File handler with rotation
        file_handler = self._create_file_handler(
            filename="therasync_pipeline.log",
            level=file_level,
            max_size=size_bytes,
            backup_count=backup_count,
        )
        root_logger.addHandler(file_handler)

        # Error file handler (errors only)
        error_handler = self._create_file_handler(
            filename="therasync_errors.log",
            level="ERROR",
            max_size=size_bytes,
            backup_count=backup_count,
        )
        root_logger.addHandler(error_handler)

        root_logger.info("Logger setup completed")
        return root_logger

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger for a specific module or component.

        Args:
            name: Logger name (typically module name).

        Returns:
            Configured logger.
        """
        if name in self._configured_loggers:
            return self._configured_loggers[name]

        logger = logging.getLogger(name)

        # If root logger is not configured, set up basic logging
        if not logging.getLogger().handlers:
            self.setup_root_logger()

        self._configured_loggers[name] = logger
        return logger

    def _create_console_handler(self, level: str) -> logging.Handler:
        """
        Create console handler with optional colors.

        Args:
            level: Logging level for console output.

        Returns:
            Configured console handler.
        """
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))

        if COLORLOG_AVAILABLE:
            # Colored formatter
            formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        else:
            # Standard formatter without colors
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        console_handler.setFormatter(formatter)
        return console_handler

    def _create_file_handler(
        self, filename: str, level: str, max_size: int, backup_count: int
    ) -> logging.Handler:
        """
        Create rotating file handler.

        Args:
            filename: Name of the log file.
            level: Logging level.
            max_size: Maximum file size in bytes.
            backup_count: Number of backup files.

        Returns:
            Configured file handler.
        """
        log_file = self.log_dir / filename

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_size, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, level.upper()))

        # Detailed formatter for file logging
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        return file_handler

    def _parse_file_size(self, size_str: str) -> int:
        """
        Parse file size string to bytes.

        Args:
            size_str: Size string (e.g., '10MB', '1GB').

        Returns:
            Size in bytes.
        """
        size_str = size_str.upper().strip()

        multipliers = {"B": 1, "KB": 1024, "MB": 1024 * 1024, "GB": 1024 * 1024 * 1024}

        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                try:
                    number = float(size_str[: -len(suffix)])
                    return int(number * multiplier)
                except ValueError:
                    break

        # Default to 10MB if parsing fails
        return 10 * 1024 * 1024

    def setup_module_logger(
        self,
        module_name: str,
        additional_file: Optional[str] = None,
        level: Optional[str] = None,
    ) -> logging.Logger:
        """
        Setup a logger for a specific module with optional dedicated file.

        Args:
            module_name: Name of the module.
            additional_file: Optional dedicated log file for this module.
            level: Optional specific log level for this module.

        Returns:
            Configured module logger.
        """
        logger = self.get_logger(module_name)

        if level:
            logger.setLevel(getattr(logging, level.upper()))

        if additional_file:
            # Add dedicated file handler for this module
            module_handler = self._create_file_handler(
                filename=additional_file,
                level=level or "DEBUG",
                max_size=self._parse_file_size("5MB"),
                backup_count=3,
            )
            logger.addHandler(module_handler)

        return logger

    def create_processing_logger(
        self, subject: str, session: str, task: str
    ) -> logging.Logger:
        """
        Create a logger for specific data processing run.

        Args:
            subject: Subject identifier.
            session: Session identifier.
            task: Task identifier.

        Returns:
            Logger for this specific processing run.
        """
        logger_name = f"processing.{subject}.{session}.{task}"
        log_filename = f"processing_{subject}_{session}_{task}.log"

        return self.setup_module_logger(
            module_name=logger_name, additional_file=log_filename, level="DEBUG"
        )

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration parameters for debugging.

        Args:
            config: Configuration dictionary to log.
        """
        logger = self.get_logger("config")
        logger.info("Configuration loaded:")

        def log_dict(d: Dict[str, Any], prefix: str = ""):
            for key, value in d.items():
                if isinstance(value, dict):
                    logger.debug(f"{prefix}{key}:")
                    log_dict(value, prefix + "  ")
                elif isinstance(value, list):
                    logger.debug(f"{prefix}{key}: {len(value)} items")
                    for i, item in enumerate(value[:3]):  # Log first 3 items
                        logger.debug(f"{prefix}  [{i}]: {item}")
                    if len(value) > 3:
                        logger.debug(f"{prefix}  ... and {len(value) - 3} more items")
                else:
                    logger.debug(f"{prefix}{key}: {value}")

        log_dict(config)

    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """
        Clean up old log files.

        Args:
            days_to_keep: Number of days to keep log files.
        """
        logger = self.get_logger("cleanup")

        import time

        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)

        removed_count = 0
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    removed_count += 1
                except OSError as e:
                    logger.warning(f"Could not remove old log file {log_file}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old log files")


# Global logger setup instance
_logger_setup: Optional[LoggerSetup] = None


def setup_logging(
    log_dir: str = "log",
    log_level: str = "INFO",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> LoggerSetup:
    """
    Setup global logging configuration.

    Args:
        log_dir: Directory for log files.
        log_level: Default logging level.
        console_level: Console logging level.
        file_level: File logging level.

    Returns:
        LoggerSetup instance.
    """
    global _logger_setup

    if _logger_setup is None:
        _logger_setup = LoggerSetup(log_dir, log_level)
        _logger_setup.setup_root_logger(console_level, file_level)

    return _logger_setup


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    global _logger_setup

    if _logger_setup is None:
        _logger_setup = setup_logging()

    return _logger_setup.get_logger(name)


if __name__ == "__main__":
    # Example usage
    setup = setup_logging()

    # Test different log levels
    logger = get_logger("test")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test module-specific logger
    module_logger = setup.setup_module_logger("test_module", "test_module.log")
    module_logger.info("Module-specific log message")

    # Test processing logger
    proc_logger = setup.create_processing_logger("g01p01", "01", "restingstate")
    proc_logger.info("Processing specific log message")
