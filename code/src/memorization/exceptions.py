"""Domain-specific exception hierarchy for configuration and protocol guardrails."""

from __future__ import annotations


class MemorizationError(Exception):
    """Base exception for the unintended memorization study."""


class ConfigurationError(MemorizationError, ValueError):
    """Raised when experiment configuration values are invalid or incompatible."""


class ProtocolViolationError(MemorizationError, RuntimeError):
    """Raised when a strict experimental protocol constraint or sanity check fails."""
