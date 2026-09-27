"""KageFrame - a statistical shadow of your DataFrame.

Profile clinical data locally and generate dummy data for code development.

V0.1 status: ``profile_dataframe`` and ``generate`` (M2-M4). Missingness, date/time
columns and ``compare`` are added in later milestones.
"""

from ._version import __version__
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
    "SCHEMA_VERSION",
    "KageFrameError",
    "KageFrameWarning",
    "ColumnType",
    "LevelMap",
    "PrivacyWarning",
    "Profile",
    "ProfileSchemaError",
    "TypeInferenceWarning",
    "TypeSpec",
    "TypeSpecError",
    "__version__",
    "generate",
    "infer_types",
    "load_profile",
    "profile_dataframe",
    "resolve_types",
]
