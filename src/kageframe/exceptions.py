class KageFrameError(Exception):
    """Base class for kageframe errors."""


class ProfileSchemaError(KageFrameError, ValueError):
    """A profile or level map does not conform to the kageframe schema."""


class TypeSpecError(KageFrameError, ValueError):
    """A ``types=`` override is invalid or inconsistent with the data."""


class KageFrameWarning(UserWarning):
    """Base class for kageframe warnings."""


class TypeInferenceWarning(KageFrameWarning):
    """A column type was inferred in a way the user should review."""


class PrivacyWarning(KageFrameWarning):
    """A column or setting may expose sensitive information."""
