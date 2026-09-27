"""Column types, type inference and ``types=`` override normalization."""

from __future__ import annotations

import datetime as _dt
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd

from .exceptions import PrivacyWarning, TypeInferenceWarning, TypeSpecError


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    BINARY = "binary"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    ID = "id"
    TEXT = "text"
    CONSTANT = "constant"
    EMPTY = "empty"
    EXCLUDED = "excluded"

    def __str__(self) -> str:
        return self.value


CATEGORICAL_TYPES = frozenset({ColumnType.BINARY, ColumnType.ORDINAL, ColumnType.NOMINAL})
TEMPORAL_TYPES = frozenset({ColumnType.DATE, ColumnType.DATETIME, ColumnType.TIME})
LATENT_TYPES = frozenset({ColumnType.NUMERIC} | CATEGORICAL_TYPES | TEMPORAL_TYPES)
# constant/empty describe the data, not the user's intent, so they cannot be requested.
OVERRIDABLE_TYPES = frozenset(set(ColumnType) - {ColumnType.CONSTANT, ColumnType.EMPTY})

DEFAULT_MAX_LEVELS = 50
DEFAULT_ORDINAL_MAX_LEVELS = 10
DATETIME_SAMPLE_SIZE = 1000
DATETIME_MATCH_RATE = 0.95

_DATE_RE = re.compile(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*$")
_DATETIME_RE = re.compile(
    r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}[ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?"
    r"(Z|[+-]\d{2}:?\d{2})?\s*$"
)
_TIME_RE = re.compile(r"^\s*([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d(\.\d+)?)?\s*$")
_SENSITIVE_NAME_RE = re.compile(
    r"((^|_)(id|mrn|patient|pt|kanja|name|shimei|address|phone|tel|email|comment|memo|note|"
    r"free)(_|$)|番号|氏名|住所|電話|備考|所見|コメント)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TypeSpec:
    """Resolved type of one column.

    ``levels`` is set only when known at resolution time: user-supplied levels,
    categories of an ordered Categorical, or sorted codes of an integer ordinal.
    ``keep_level_names=None`` means the default (pseudonymize string levels).
    """

    type: ColumnType
    levels: tuple[Any, ...] | None = None
    keep_level_names: bool | None = None
    source: Literal["override", "inferred"] = "inferred"
    reason: str = ""


@dataclass(frozen=True)
class TypeWarning:
    column: str
    code: str
    message: str
    category: type[Warning] = field(default=TypeInferenceWarning, compare=False)

    def to_record(self) -> dict[str, str]:
        return {"column": self.column, "code": self.code, "message": self.message}


@dataclass
class TypeResolution:
    specs: dict[str, TypeSpec]
    warnings: list[TypeWarning]

    def emit_warnings(self, stacklevel: int = 3) -> None:
        for w in self.warnings:
            warnings.warn(f"[{w.column}] {w.message}", w.category, stacklevel=stacklevel)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _nonmissing(s: pd.Series) -> pd.Series:
    return s[s.notna()]


def _is_integer_valued(values: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(values):
        return False
    if pd.api.types.is_integer_dtype(values):
        return True
    if pd.api.types.is_float_dtype(values):
        arr = values.to_numpy(dtype=float)
        return bool(np.all(np.isfinite(arr)) and np.all(arr == np.round(arr)))
    return False


def _all_instances(values: pd.Series, types: type | tuple[type, ...]) -> bool:
    sample = values.iloc[:DATETIME_SAMPLE_SIZE]
    return len(sample) > 0 and all(isinstance(v, types) for v in sample)


def _is_midnight(values: pd.Series) -> bool:
    dt = values.dt
    return bool(
        ((dt.hour == 0) & (dt.minute == 0) & (dt.second == 0) & (dt.microsecond == 0)
         & (dt.nanosecond == 0)).all()
    )


def _match_rate(sample: pd.Series, *patterns: re.Pattern[str]) -> float:
    hits = sample.map(lambda v: any(p.match(v) for p in patterns))
    return float(hits.mean()) if len(sample) else 0.0


def _infer_string_temporal(values: pd.Series) -> ColumnType | None:
    sample = values.iloc[:DATETIME_SAMPLE_SIZE].astype(str)
    if _match_rate(sample, _TIME_RE) >= DATETIME_MATCH_RATE:
        return ColumnType.TIME
    if _match_rate(sample, _DATE_RE, _DATETIME_RE) < DATETIME_MATCH_RATE:
        return None
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    if parsed.notna().mean() < DATETIME_MATCH_RATE:
        return None
    return ColumnType.DATE if _is_midnight(parsed.dropna()) else ColumnType.DATETIME


def _looks_like_integer_id(values: pd.Series) -> bool:
    n = len(values)
    n_unique = values.nunique()
    if n_unique <= DEFAULT_MAX_LEVELS or n_unique / n < 0.99:
        return False
    arr = values.to_numpy(dtype=float)
    span = arr.max() - arr.min() + 1
    return bool(values.is_monotonic_increasing or span / n_unique <= 1.1)


def _string_profile(values: pd.Series) -> tuple[int, float, float, float]:
    strings = values.astype(str)
    n_unique = strings.nunique()
    unique_ratio = n_unique / len(strings)
    avg_len = float(strings.str.len().mean())
    frac_space = float(strings.str.contains(r"\s", regex=True).mean())
    return n_unique, unique_ratio, avg_len, frac_space


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def infer_column_type(
    series: pd.Series,
    *,
    max_levels: int = DEFAULT_MAX_LEVELS,
    ordinal_max_levels: int = DEFAULT_ORDINAL_MAX_LEVELS,
) -> tuple[TypeSpec, list[TypeWarning]]:
    """Infer the type of one column following the rules in the V0.1 plan (§2.2)."""
    name = str(series.name)
    warns: list[TypeWarning] = []
    values = _nonmissing(series)
    dtype = series.dtype

    def spec(t: ColumnType, reason: str, levels: tuple[Any, ...] | None = None) -> TypeSpec:
        return TypeSpec(type=t, levels=levels, source="inferred", reason=reason)

    if len(values) == 0:
        return spec(ColumnType.EMPTY, "all values missing"), warns
    try:
        n_unique = values.nunique()
    except TypeError:
        values = values.astype(str)
        n_unique = values.nunique()
    if n_unique == 1:
        return spec(ColumnType.CONSTANT, "single non-missing value"), warns

    if pd.api.types.is_bool_dtype(dtype) or _all_instances(values, (bool, np.bool_)):
        return spec(ColumnType.BINARY, "boolean values"), warns

    if isinstance(dtype, pd.CategoricalDtype):
        if dtype.ordered:
            return spec(ColumnType.ORDINAL, "ordered Categorical",
                        tuple(dtype.categories.tolist())), warns
        if n_unique == 2:
            return spec(ColumnType.BINARY, "Categorical with 2 observed levels"), warns
        return spec(ColumnType.NOMINAL, "unordered Categorical"), warns

    if pd.api.types.is_datetime64_any_dtype(dtype):
        t = ColumnType.DATE if _is_midnight(values) else ColumnType.DATETIME
        return spec(t, "datetime64 dtype"), warns
    if pd.api.types.is_timedelta64_dtype(dtype) or _all_instances(values, _dt.time):
        return spec(ColumnType.TIME, "time-of-day values"), warns
    if _all_instances(values, _dt.datetime):
        parsed = pd.to_datetime(values)
        t = ColumnType.DATE if _is_midnight(parsed) else ColumnType.DATETIME
        return spec(t, "datetime objects"), warns
    if _all_instances(values, _dt.date):
        return spec(ColumnType.DATE, "date objects"), warns

    if pd.api.types.is_numeric_dtype(dtype):
        if _is_integer_valued(values):
            unique = set(values.unique().tolist())
            if unique == {0, 1}:
                return spec(ColumnType.BINARY, "integer values {0, 1}"), warns
            if _looks_like_integer_id(values):
                warns.append(TypeWarning(name, "id_like_excluded",
                                         "integer column looks like an identifier; excluded "
                                         "and regenerated. Override with types= if wrong.",
                                         PrivacyWarning))
                return spec(ColumnType.ID, "unique, (near) consecutive integers"), warns
            if n_unique <= ordinal_max_levels:
                levels = tuple(sorted(unique))
                warns.append(TypeWarning(
                    name, "integer_code_as_ordinal",
                    f"integer column with {n_unique} distinct values inferred as ordinal; "
                    "specifying the type explicitly via types= is recommended."))
                return spec(ColumnType.ORDINAL, f"integer codes (<= {ordinal_max_levels} levels)",
                            levels), warns
        return spec(ColumnType.NUMERIC, "numeric dtype"), warns

    # object / string columns
    values = values.astype(str) if not _all_instances(values, str) else values
    temporal = _infer_string_temporal(values)
    if temporal is not None:
        return spec(temporal, "strings parseable as date/time"), warns

    n_unique, unique_ratio, avg_len, frac_space = _string_profile(values)
    sensitive_name = bool(_SENSITIVE_NAME_RE.search(name))
    if n_unique <= max_levels:
        if avg_len > 50 and unique_ratio >= 0.5:
            warns.append(TypeWarning(name, "free_text_excluded",
                                     "long, mostly unique strings look like free text; "
                                     "excluded from the output.", PrivacyWarning))
            return spec(ColumnType.TEXT, "long mostly-unique strings"), warns
        if sensitive_name:
            warns.append(TypeWarning(
                name, "sensitive_column_name",
                "column name suggests identifying information but the column was profiled as "
                "categorical; review it or exclude it via types=.", PrivacyWarning))
        t = ColumnType.BINARY if n_unique == 2 else ColumnType.NOMINAL
        return spec(t, f"{n_unique} distinct strings"), warns

    if unique_ratio >= 0.9 and avg_len <= 32 and frac_space < 0.1:
        warns.append(TypeWarning(name, "id_like_excluded",
                                 "mostly unique short strings look like an identifier; excluded "
                                 "and regenerated.", PrivacyWarning))
        return spec(ColumnType.ID, "mostly unique short strings"), warns
    warns.append(TypeWarning(
        name, "free_text_excluded",
        f"{n_unique} distinct strings exceed max_levels={max_levels}; treated as free text and "
        "excluded from the output.", PrivacyWarning))
    return spec(ColumnType.TEXT, f"more than {max_levels} distinct strings"), warns


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------

_OVERRIDE_KEYS = {"type", "levels", "keep_level_names"}


def _parse_type_name(value: Any, column: str) -> ColumnType:
    try:
        t = ColumnType(value.value if isinstance(value, ColumnType) else str(value).lower())
    except ValueError:
        allowed = ", ".join(sorted(t.value for t in OVERRIDABLE_TYPES))
        raise TypeSpecError(f"[{column}] unknown type {value!r}; allowed: {allowed}") from None
    if t not in OVERRIDABLE_TYPES:
        raise TypeSpecError(f"[{column}] type {t.value!r} is derived from the data and "
                            "cannot be requested")
    return t


def _parse_override(column: str, value: Any) -> TypeSpec:
    if isinstance(value, Mapping):
        unknown = set(value) - _OVERRIDE_KEYS
        if unknown:
            raise TypeSpecError(f"[{column}] unknown override keys: {sorted(unknown)}")
        if "type" not in value:
            raise TypeSpecError(f"[{column}] override dict needs a 'type' key")
        t = _parse_type_name(value["type"], column)
        levels = value.get("levels")
        keep = value.get("keep_level_names")
    else:
        t, levels, keep = _parse_type_name(value, column), None, None

    if levels is not None:
        if t not in CATEGORICAL_TYPES:
            raise TypeSpecError(f"[{column}] 'levels' is only valid for binary/ordinal/nominal")
        if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
            raise TypeSpecError(f"[{column}] 'levels' must be a list")
        levels = tuple(levels)
        if len(set(levels)) != len(levels):
            raise TypeSpecError(f"[{column}] 'levels' contains duplicates")
        if t is ColumnType.BINARY and len(levels) != 2:
            raise TypeSpecError(f"[{column}] binary 'levels' must have exactly 2 entries")
        if len(levels) < 2:
            raise TypeSpecError(f"[{column}] 'levels' must have at least 2 entries")
    if keep is not None and not isinstance(keep, bool):
        raise TypeSpecError(f"[{column}] 'keep_level_names' must be a bool")
    return TypeSpec(type=t, levels=levels, keep_level_names=keep, source="override",
                    reason="user override")


def _level_key(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return ("num", float(value))
    return ("str", str(value))


def _check_override_against_data(column: str, spec: TypeSpec, series: pd.Series) -> TypeSpec:
    values = _nonmissing(series)
    t = spec.type
    if t in CATEGORICAL_TYPES and len(values):
        observed = values.unique().tolist()
        if spec.levels is not None:
            allowed = {_level_key(v) for v in spec.levels}
            missing = [v for v in observed if _level_key(v) not in allowed]
            if missing:
                shown = ", ".join(repr(v) for v in missing[:5])
                raise TypeSpecError(f"[{column}] values not listed in 'levels': {shown}")
        elif t is ColumnType.BINARY and len(observed) > 2:
            raise TypeSpecError(f"[{column}] binary override but {len(observed)} distinct values")
        elif t is ColumnType.ORDINAL:
            dtype = series.dtype
            if isinstance(dtype, pd.CategoricalDtype) and dtype.ordered:
                spec = TypeSpec(t, tuple(dtype.categories.tolist()), spec.keep_level_names,
                                spec.source, spec.reason)
            elif pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                spec = TypeSpec(t, tuple(sorted(observed)), spec.keep_level_names,
                                spec.source, spec.reason)
            else:
                raise TypeSpecError(f"[{column}] ordinal override on non-numeric values needs "
                                    "explicit 'levels' (the order is never guessed)")
    if t is ColumnType.NUMERIC and len(values):
        if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
            series.dtype
        ):
            converted = pd.to_numeric(values, errors="coerce")
            if converted.isna().any():
                raise TypeSpecError(f"[{column}] numeric override but values are not numeric")
    return spec


def _normalize_keep_level_names(
    keep_level_names: Sequence[str] | Literal["all"] | None, columns: Sequence[str]
) -> set[str]:
    if keep_level_names is None:
        return set()
    if isinstance(keep_level_names, str):
        if keep_level_names != "all":
            raise TypeSpecError("keep_level_names must be a list of column names or 'all'")
        return set(columns)
    names = set(keep_level_names)
    unknown = names - set(columns)
    if unknown:
        raise TypeSpecError(f"keep_level_names refers to unknown columns: {sorted(unknown)}")
    return names


def _check_columns(df: pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    bad = [c for c in columns if not isinstance(c, str)]
    if bad:
        raise TypeSpecError(f"column names must be strings; got {bad[:5]!r}")
    dupes = sorted({c for c in columns if columns.count(c) > 1})
    if dupes:
        raise TypeSpecError(f"duplicate column names: {dupes}")
    return columns


def resolve_types(
    df: pd.DataFrame,
    types: Mapping[str, Any] | None = None,
    *,
    keep_level_names: Sequence[str] | Literal["all"] | None = None,
    max_levels: int = DEFAULT_MAX_LEVELS,
    ordinal_max_levels: int = DEFAULT_ORDINAL_MAX_LEVELS,
) -> TypeResolution:
    """Resolve the type of every column: overrides first, inference otherwise.

    Overrides are validated against the data; inferred types are accompanied by
    warnings the user should review (integer codes, identifiers, free text).
    """
    columns = _check_columns(df)
    types = dict(types or {})
    unknown = set(types) - set(columns)
    if unknown:
        raise TypeSpecError(f"types refers to unknown columns: {sorted(unknown)}")
    keep = _normalize_keep_level_names(keep_level_names, columns)

    specs: dict[str, TypeSpec] = {}
    warns: list[TypeWarning] = []
    for name in columns:
        series = df[name]
        if name in types:
            spec = _check_override_against_data(name, _parse_override(name, types[name]), series)
        else:
            spec, col_warns = infer_column_type(series, max_levels=max_levels,
                                                ordinal_max_levels=ordinal_max_levels)
            warns.extend(col_warns)
        if spec.keep_level_names is None and name in keep:
            spec = TypeSpec(spec.type, spec.levels, True, spec.source, spec.reason)
        specs[name] = spec

    if keep_level_names == "all":
        warns.append(TypeWarning("*", "level_names_kept",
                                 "keep_level_names='all': real level names will be stored in "
                                 "the profile.", PrivacyWarning))
    return TypeResolution(specs=specs, warnings=warns)


def infer_types(
    df: pd.DataFrame,
    types: Mapping[str, Any] | None = None,
    *,
    max_levels: int = DEFAULT_MAX_LEVELS,
    ordinal_max_levels: int = DEFAULT_ORDINAL_MAX_LEVELS,
) -> dict[str, str]:
    """Return ``{column: type}`` as it would be used by ``profile_dataframe``.

    Review the result and pass corrections back via ``types=``.
    """
    resolution = resolve_types(df, types, max_levels=max_levels,
                               ordinal_max_levels=ordinal_max_levels)
    resolution.emit_warnings()
    return {name: spec.type.value for name, spec in resolution.specs.items()}
