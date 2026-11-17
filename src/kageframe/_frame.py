"""Polars helpers shared by profiling, generation and comparison."""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any

import numpy as np
import polars as pl

from .exceptions import TypeSpecError

_DATETIME_PARTS_RE = re.compile(
    r"^\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"
    r"(?:[ T](\d{1,2}):(\d{2})(?::(\d{2})(\.\d+)?)?)?"
    r"\s*(Z|[+-]\d{2}:?\d{2})?\s*$"
)
_TIME_PARTS_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2})(\.\d+)?)?\s*$")


def as_frame(df: Any) -> pl.DataFrame:
    """Return ``df`` as an eager Polars DataFrame (a LazyFrame is collected)."""
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"expected a polars DataFrame, got {type(df).__module__}."
                        f"{type(df).__name__}; convert it first, e.g. pl.from_pandas(df)")
    return df


def prepare(df: Any) -> pl.DataFrame:
    """Collect, check the column names and turn float NaN into null (both are missing)."""
    df = as_frame(df)
    floats = [c for c, t in df.schema.items() if t.is_float()]
    if floats:
        df = df.with_columns(pl.col(floats).fill_nan(None))
    objects = [c for c, t in df.schema.items() if t == pl.Object]
    if objects:
        df = df.with_columns([to_strings(df[c]) for c in objects])
    return df


def nonmissing(s: pl.Series) -> pl.Series:
    return s.drop_nulls()


def dtype_name(dtype: pl.DataType) -> str:
    """Stable name of a dtype for the profile (never includes category values)."""
    base = dtype.base_type()
    if base in (pl.Datetime, pl.Duration):
        return f"{base.__name__}[{dtype.time_unit}]"  # type: ignore[attr-defined]
    if base is pl.Decimal:
        return "Decimal"
    if base in (pl.List, pl.Array, pl.Struct):
        return base.__name__
    return base.__name__


_SIMPLE_DTYPES: dict[str, pl.DataType] = {
    name: getattr(pl, name)()
    for name in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
                 "Float32", "Float64", "Boolean", "String", "Date", "Time", "Null")
}
_UNIT_RE = re.compile(r"^(Datetime|Duration)\[(ms|us|ns)\]$")


def parse_dtype(name: str) -> pl.DataType | None:
    """Inverse of :func:`dtype_name` for the dtypes generation can restore (else None)."""
    if name in _SIMPLE_DTYPES:
        return _SIMPLE_DTYPES[name]
    if name == "Categorical":
        return pl.Categorical()
    m = _UNIT_RE.match(name)
    if m:
        return (pl.Datetime if m.group(1) == "Datetime" else pl.Duration)(m.group(2))
    return None


def to_strings(s: pl.Series) -> pl.Series:
    """Values as strings (null stays null)."""
    if s.dtype == pl.String:
        return s
    if s.dtype.base_type() in (pl.Categorical, pl.Enum):
        return s.cast(pl.String)
    if s.dtype == pl.Object:
        return pl.Series(s.name, [None if v is None else str(v) for v in s.to_list()],
                         dtype=pl.String)
    try:
        return s.cast(pl.String)
    except pl.exceptions.PolarsError:
        return pl.Series(s.name, [None if v is None else str(v) for v in s.to_list()],
                         dtype=pl.String)


def _parse_datetime(text: str) -> tuple[_dt.datetime, bool] | None:
    m = _DATETIME_PARTS_RE.match(text)
    if not m:
        return None
    y, mo, d, h, mi, sec, frac, off = m.groups()
    try:
        value = _dt.datetime(int(y), int(mo), int(d), int(h or 0), int(mi or 0),
                             int(sec or 0), int(round(float(frac or 0) * 1e6)) % 1_000_000)
    except ValueError:
        return None
    if off:
        if off == "Z":
            delta = _dt.timedelta(0)
        else:
            sign = -1 if off[0] == "-" else 1
            digits = off[1:].replace(":", "")
            delta = sign * _dt.timedelta(hours=int(digits[:2]), minutes=int(digits[2:]))
        return value - delta, True
    return value, False


def parse_datetime_strings(s: pl.Series) -> tuple[pl.Series, str | None]:
    """Parse ``YYYY-MM-DD[ HH:MM[:SS[.f]]][offset]`` strings (also ``/`` and ``.``).

    Unparseable values become null. Values with a UTC offset are converted to UTC; if any
    value had an offset the time zone ``"UTC"`` is returned with naive UTC wall times.
    """
    strings = to_strings(s)
    uniques = strings.drop_nulls().unique().to_list()
    parsed = {u: _parse_datetime(u) for u in uniques}
    has_offset = any(v is not None and v[1] for v in parsed.values())
    keys = list(parsed)
    values = [None if parsed[k] is None else parsed[k][0] for k in keys]
    out = strings.replace_strict(keys, pl.Series(values, dtype=pl.Datetime("us")),
                                 default=None, return_dtype=pl.Datetime("us"))
    return out.alias(s.name), ("UTC" if has_offset else None)


def parse_time_string(text: str) -> int | None:
    """Whole seconds from midnight of ``H:MM[:SS[.f]]``, or None."""
    m = _TIME_PARTS_RE.match(text)
    if not m or int(m.group(1)) >= 24 or int(m.group(2)) >= 60 \
            or int(m.group(3) or 0) >= 60:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0)


def float_array(s: pl.Series) -> np.ndarray:
    """Float values with NaN where missing."""
    return s.cast(pl.Float64, strict=False).fill_null(np.nan).to_numpy().astype(float)


def check_columns(df: pl.DataFrame) -> list[str]:
    columns = list(df.columns)
    dupes = sorted({c for c in columns if columns.count(c) > 1})
    if dupes:  # Polars forbids this; kept for clarity of the error
        raise TypeSpecError(f"duplicate column names: {dupes}")
    return columns
