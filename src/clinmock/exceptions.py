class ClinmockError(Exception):
    """Base class for clinmock errors."""


class ProfileSchemaError(ClinmockError, ValueError):
    """A profile or level map does not conform to the clinmock schema."""


class TypeSpecError(ClinmockError, ValueError):
    """A ``types=`` override is invalid or inconsistent with the data."""


class ClinmockWarning(UserWarning):
    """Base class for clinmock warnings."""


class TypeInferenceWarning(ClinmockWarning):
    """A column type was inferred in a way the user should review."""


class PrivacyWarning(ClinmockWarning):
    """A column or setting may expose sensitive information."""
