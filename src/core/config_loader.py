"""
Configuration loader for Therasync Pipeline.

This module provides utilities for loading and validating configuration files
in YAML format. It includes schema validation and environment variable support.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import jsonschema
from jsonschema import validate


logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""

    pass


class ConfigLoader:
    """
    Configuration loader with schema validation and environment variable support.

    This class handles loading YAML configuration files, validating them against
    a schema, and providing easy access to configuration parameters.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        self.config_path = self._resolve_config_path(config_path)
        self.config: Dict[str, Any] = {}
        self._schema = self._load_schema()

        # Automatically load the configuration
        self.load_config()

    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """
        Resolve the configuration file path.

        Args:
            config_path: User-provided path or None for default.

        Returns:
            Resolved path to configuration file.
        """
        if config_path is None:
            # Default to config/config.yaml relative to project root
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            config_path = project_root / "config" / "config.yaml"

        return Path(config_path)

    def _load_schema(self) -> Dict[str, Any]:
        """
        Load the configuration schema for validation.

        Returns:
            JSON schema dictionary for configuration validation.
        """
        # Basic schema for configuration validation
        schema = {
            "type": "object",
            "required": ["study", "paths", "moments", "physio"],
            "properties": {
                "study": {
                    "type": "object",
                    "required": ["name", "version"],
                    "properties": {
                        "name": {"type": "string"},
                        "version": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
                "paths": {
                    "type": "object",
                    "required": ["rawdata", "derivatives"],
                    "properties": {
                        "rawdata": {"type": "string"},
                        "derivatives": {"type": "string"},
                        "logs": {"type": "string"},
                    },
                },
                "moments": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "duration_expected": {"type": "number"},
                        },
                    },
                },
                "physio": {
                    "type": "object",
                    "properties": {
                        "bvp": {"type": "object"},
                        "eda": {"type": "object"},
                        "hr": {"type": "object"},
                    },
                },
            },
        }
        return schema

    def load_config(self) -> Dict[str, Any]:
        """
        Load and validate the configuration file.

        Returns:
            Loaded and validated configuration dictionary.

        Raises:
            ConfigError: If configuration file is invalid or missing.
        """
        try:
            # Check if file exists
            if not self.config_path.exists():
                raise ConfigError(f"Configuration file not found: {self.config_path}")

            # Load YAML file
            with open(self.config_path, "r", encoding="utf-8") as file:
                self.config = yaml.safe_load(file)

            # Validate against schema
            self._validate_config()

            # Process environment variables
            self._process_environment_variables()

            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self.config

        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML configuration: {e}")
        except jsonschema.ValidationError as e:
            raise ConfigError(f"Configuration validation error: {e.message}")
        except Exception as e:
            raise ConfigError(f"Unexpected error loading configuration: {e}")

    def _validate_config(self) -> None:
        """
        Validate the loaded configuration against the schema.

        Raises:
            jsonschema.ValidationError: If configuration is invalid.
        """
        validate(instance=self.config, schema=self._schema)

    def _process_environment_variables(self) -> None:
        """
        Process environment variables in configuration values.

        Replaces ${ENV_VAR} patterns with environment variable values.
        """

        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str):
                # Simple environment variable substitution
                if obj.startswith("${") and obj.endswith("}"):
                    env_var = obj[2:-1]
                    return os.getenv(env_var, obj)
                return obj
            else:
                return obj

        self.config = replace_env_vars(self.config)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'physio.bvp.sampling_rate').
            default: Default value if key is not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value: Any = self.config

        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default

    def get_moments(self) -> list:
        """
        Get the list of configured moments/tasks.

        Returns:
            List of moment configurations.
        """
        if isinstance(self.config, dict):
            return self.config.get("moments", [])
        return []

    def get_moment_names(self) -> list:
        """
        Get the list of moment names.

        Returns:
            List of moment names.
        """
        return [moment["name"] for moment in self.get_moments()]

    def get_physio_config(self, signal_type: str) -> Dict[str, Any]:
        """
        Get physiological signal configuration.

        Args:
            signal_type: Type of signal ('bvp', 'eda', 'hr').

        Returns:
            Configuration dictionary for the specified signal type.
        """
        if isinstance(self.config, dict):
            physio_config = self.config.get("physio", {})
            if isinstance(physio_config, dict):
                return physio_config.get(signal_type, {})
        return {}

    def get_bids_config(self) -> Dict[str, Any]:
        """
        Get BIDS configuration settings.

        Returns:
            BIDS configuration dictionary.
        """
        if isinstance(self.config, dict):
            return self.config.get("bids", {})
        return {}

    def get_paths(self) -> Dict[str, str]:
        """
        Get path configurations.

        Returns:
            Dictionary of configured paths.
        """
        if isinstance(self.config, dict):
            return self.config.get("paths", {})
        return {}

    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the current configuration to a file.

        Args:
            output_path: Path to save the configuration. If None, overwrites original.
        """
        if output_path is None:
            output_path = self.config_path
        else:
            output_path = Path(output_path)

        try:
            with open(output_path, "w", encoding="utf-8") as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            raise ConfigError(f"Error saving configuration: {e}")


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to configuration file or None for default.

    Returns:
        Loaded configuration dictionary.
    """
    loader = ConfigLoader(config_path)
    return loader.load_config()


if __name__ == "__main__":
    # Example usage
    try:
        config = load_config()
        print("Configuration loaded successfully!")
        print(f"Study name: {config['study']['name']}")
        print(f"Available moments: {[m['name'] for m in config['moments']]}")
    except ConfigError as e:
        print(f"Configuration error: {e}")
