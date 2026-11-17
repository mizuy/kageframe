"""date / datetime / time columns: numeric parts (days, seconds) and their inverse."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import polars as pl

from .._frame import dtype_name, parse_datetime_strings, parse_time_string, to_strings
from ..schema import DateColumn, DatetimeColumn, Quantiles, TemporalPart, TimeColumn, WarningRecord
from ..types import ColumnType
from . import ColumnResult
from .numeric import quantile_grid, score_feature, too_few_values

SECONDS_PER_DAY = 86400
_EPOCH = np.datetime64("1970-01-01", "D")
_DATE_SEP_RE = re.compile(r"^\s*\d{4}([-/.])")


def _is_string_source(series: pl.Series) -> bool:
    return series.dtype.base_type() in (pl.String, pl.Categorical, pl.Enum)


def to_timestamps(series: pl.Series) -> tuple[pl.Series, str | None]:
    """Naive wall-clock ``Datetime[us]`` values (null = missing) and the time zone name."""
    dtype = series.dtype
    base = dtype.base_type()
    if base is pl.Datetime:
        tz = dtype.time_zone
        s = series.dt.replace_time_zone(None) if tz is not None else series
        return s.cast(pl.Datetime("us")), tz
    if base is pl.Date:
        return series.cast(pl.Datetime("us")), None
    return parse_datetime_strings(series)


def date_time_parts(ts: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Days since 1970-01-01 and whole seconds from midnight, NaN where missing."""
    us = ts.cast(pl.Datetime("us")).to_physical()
    missing = us.is_null().to_numpy()
    us_arr = us.fill_null(0).to_numpy().astype(np.int64)
    day_us = SECONDS_PER_DAY * 10**6
    days = np.floor_divide(us_arr, day_us)
    secs = (us_arr - days * day_us) // 10**6
    days, secs = days.astype(float), secs.astype(float)
    days[missing] = np.nan
    secs[missing] = np.nan
    return days, secs


def time_seconds(series: pl.Series) -> np.ndarray:
    """Seconds from midnight of Time, Duration, Datetime or ``H:MM[:SS]`` string values."""
    base = series.dtype.base_type()
    if base is pl.Time:
        ns = series.to_physical().cast(pl.Float64).fill_null(np.nan).to_numpy()
        return np.floor(ns / 1e9)
    if base is pl.Duration:
        us = series.cast(pl.Duration("us")).to_physical().cast(pl.Float64)
        return np.floor(us.fill_null(np.nan).to_numpy() / 1e6)
    if base in (pl.Datetime, pl.Date):
        return date_time_parts(to_timestamps(series)[0])[1]
    strings = to_strings(series)
    uniques = strings.drop_nulls().unique().to_list()
    secs = {u: parse_time_string(u) for u in uniques}
    out = strings.replace_strict(list(secs), list(secs.values()), default=None,
                                 return_dtype=pl.Float64)
    return out.fill_null(np.nan).to_numpy().astype(float)


def detect_resolution(secs: np.ndarray) -> int:
    s = secs[~np.isnan(secs)]
    for res in (3600, 60):
        if np.all(s % res == 0):
            return res
    return 1


def _date_format(first: str) -> str:
    m = _DATE_SEP_RE.match(first)
    sep = m.group(1) if m else "-"
    return f"%Y{sep}%m{sep}%d"


def _string_format(series: pl.Series, t: ColumnType, resolution: int | None) -> str | None:
    if not _is_string_source(series):
        return None
    values = to_strings(series).drop_nulls().head(1000)
    with_seconds = bool(values.str.contains(r"\d:\d{2}:\d{2}").any()) \
        or (resolution is not None and resolution < 60)
    time_fmt = "%H:%M:%S" if with_seconds else "%H:%M"
    if t is ColumnType.TIME:
        return time_fmt
    date_fmt = _date_format(values[0])
    if t is ColumnType.DATE:
        return date_fmt
    joiner = "T" if values.str.contains(r"\dT\d").any() else " "
    return f"{date_fmt}{joiner}{time_fmt}"


def _part(x_all: np.ndarray, k: int, options: dict[str, Any], resolution: int | None
          ) -> tuple[TemporalPart | None, np.ndarray, np.ndarray]:
    feature, cells = score_feature(x_all)
    x = x_all[~np.isnan(x_all)]
    if len(x) < 2 * k:
        return None, feature, cells
    grid: Quantiles = quantile_grid(x, k, options["quantile_grid"])
    step = 1 if resolution is None else resolution
    lo = float(np.round(grid.values[0] / step) * step)
    hi = float(np.round(grid.values[-1] / step) * step)
    return TemporalPart(lo, hi, grid, resolution), feature, cells


def profile_temporal(name: str, series: pl.Series, t: ColumnType, *, n_rows: int,
                     options: dict[str, Any]) -> ColumnResult:
    """Profile a date, datetime or time column (min/max and grids follow the k rule)."""
    k = options["min_tail_count"]
    warnings: list[WarningRecord] = []
    common: dict[str, Any] = dict(name=name, type=t, source_dtype=dtype_name(series.dtype))

    if t is ColumnType.TIME:
        secs = time_seconds(series)
        n = int((~np.isnan(secs)).sum())
        res = detect_resolution(secs)
        part, feature, cells = _part(secs, k, options, res)
        if part is None:
            warnings.append(too_few_values(name, k))
            part = TemporalPart(None, None, None, res)
        column = TimeColumn(**common, n_nonmissing=n, missing_rate=1 - n / n_rows,
                            time_part=part, output_format=_string_format(series, t, res))
        return ColumnResult(column, [feature], [cells], warnings)

    ts, tz = to_timestamps(series)
    days, secs = date_time_parts(ts)
    n = int((~np.isnan(days)).sum())
    date_part, f_date, c_date = _part(days, k, options, None)
    if t is ColumnType.DATE:
        if date_part is None:
            warnings.append(too_few_values(name, k))
            date_part = TemporalPart(None, None, None)
        column = DateColumn(**common, n_nonmissing=n, missing_rate=1 - n / n_rows,
                            date_part=date_part, output_format=_string_format(series, t, None))
        return ColumnResult(column, [f_date], [c_date], warnings)

    res = detect_resolution(secs)
    time_part, f_time, c_time = _part(secs, k, options, res)
    if date_part is None or time_part is None:
        warnings.append(too_few_values(name, k))
        date_part = TemporalPart(None, None, None)
        time_part = TemporalPart(None, None, None, res)
    column = DatetimeColumn(**common, n_nonmissing=n, missing_rate=1 - n / n_rows,
                            date_part=date_part, time_part=time_part, tz=tz,
                            output_format=_string_format(series, t, res))
    return ColumnResult(column, [f_date, f_time], [c_date, c_time], warnings)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def sample_part(part: TemporalPart, u: np.ndarray, step: int) -> np.ndarray | None:
    """Values of ``part`` at probabilities ``u``, rounded to multiples of ``step``."""
    if part.grid is None:
        return None
    x = np.interp(u, part.grid.probs, part.grid.values)
    return np.round(x / step) * step


def build_dates(days: np.ndarray) -> np.ndarray:
    return _EPOCH + days.astype(np.int64).astype("timedelta64[D]")


def datetime_values(days: np.ndarray, secs: np.ndarray) -> np.ndarray:
    base = build_dates(days).astype("datetime64[s]")
    return base + secs.astype(np.int64).astype("timedelta64[s]")


def clip_seconds(secs: np.ndarray, resolution: int) -> np.ndarray:
    return np.clip(secs, 0, SECONDS_PER_DAY - resolution)


def format_times(secs: np.ndarray, fmt: str | None, source_dtype: str) -> pl.Series:
    """Seconds from midnight as Duration (Duration source), ``fmt`` strings or Time."""
    s = secs.astype(np.int64)
    if source_dtype.startswith("Duration"):
        return pl.Series(s * 10**6, dtype=pl.Int64).cast(pl.Duration("us"))
    if fmt is None:
        return pl.Series(s * 10**9, dtype=pl.Int64).cast(pl.Time)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    with_seconds = "%S" in fmt
    return pl.Series([f"{a:02d}:{b:02d}:{c:02d}" if with_seconds else f"{a:02d}:{b:02d}"
                      for a, b, c in zip(h, m, sec, strict=True)], dtype=pl.String)
