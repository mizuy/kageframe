"""KageFrame - a statistical shadow of your DataFrame.

Profile a (clinical) DataFrame locally into aggregate statistics, then generate dummy data
with the same columns, similar marginal distributions, missing rates and rough pairwise
dependence, so that analysis code can be written without access to the real rows.

Main entry points:

- :func:`profile_dataframe` - DataFrame -> :class:`Profile` (JSON-serializable)
- :func:`generate` - :class:`Profile` -> dummy DataFrame
- :func:`compare` - how closely a dummy follows a profile (:class:`CompareReport`)
- :func:`infer_types` / :func:`resolve_types` - inspect the inferred column types
"""

from ._version import __version__
from .compare import DEFAULT_TOLERANCES, CheckResult, CompareReport, compare
from .exceptions import (
    KageFrameError,
    KageFrameWarning,
    PrivacyWarning,
    ProfileSchemaError,
    TypeInferenceWarning,
    TypeSpecError,
)
from .generate import generate
from .profile import profile_dataframe
from .schema import SCHEMA_VERSION, LevelMap, Profile, load_profile
from .types import ColumnType, TypeSpec, infer_types, resolve_types

__all__ = [
    "DEFAULT_TOLERANCES",
    "SCHEMA_VERSION",
    "CheckResult",
    "ColumnType",
    "CompareReport",
    "KageFrameError",
    "KageFrameWarning",
    "LevelMap",
    "PrivacyWarning",
    "Profile",
    "ProfileSchemaError",
    "TypeInferenceWarning",
    "TypeSpec",
    "TypeSpecError",
    "__version__",
    "compare",
    "generate",
    "infer_types",
    "load_profile",
    "profile_dataframe",
    "resolve_types",
]
