"""
PRKit exception hierarchy.

All package-specific exceptions derive from PRKitError so callers can catch
the entire family with a single ``except PRKitError`` clause while still
catching individual subtypes for finer-grained handling.

Subclasses dual-inherit the closest matching builtin so that existing call
sites which assert on builtins (e.g. ``except ValueError``) continue to work
without modification.
"""


class PRKitError(Exception):
    """Base class for all PRKit-specific exceptions."""


class UnknownModelError(PRKitError, ValueError):
    """Raised when a model name does not match any registered provider."""


class ModelClientError(PRKitError, RuntimeError):
    """Raised when a provider API call fails in a way that has useful context."""


class ConfigError(PRKitError, ValueError):
    """Raised for misconfigured or missing environment / config values."""


class DatasetError(PRKitError, RuntimeError):
    """Raised for dataset loading or download failures."""
