"""Project-level exception types."""


class GPCRRingGenError(Exception):
    """Base class for expected project errors."""


class ConfigurationError(GPCRRingGenError):
    """Raised when configuration cannot be loaded or validated."""


class RegistryError(GPCRRingGenError):
    """Raised for invalid registry operations."""


class PipelineError(GPCRRingGenError):
    """Raised when a software pipeline cannot complete."""
