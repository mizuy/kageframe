"""date / datetime / time columns: numeric parts (days, seconds) and their inverse."""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any

import numpy as np
import pandas as pd

from ..schema import DateColumn, DatetimeColumn, Quantiles, TemporalPart, TimeColumn, WarningRecord
from ..types import ColumnType
from . import ColumnResult
from .numeric import quantile_grid, score_feature, too_few_values

SECONDS_PER_DAY = 86400
_EPOCH = np.datetime64("1970-01-01", "D")
_TIME_PARTS_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2})(\.\d+)?)?\s*$")
_DATE_SEP_RE = re.compile(r"^\s*\d{4}([-/.])")


def _is_string_source(series: pd.Series) -> bool:
    values = series.dropna()
    return len(values) > 0 and isinstance(values.iloc[0], str)


def to_timestamps(series: pd.Series) -> tuple[pd.Series, str | None]:
    """Naive wall-clock timestamps (NaT = missing) and the original time zone name."""
    s = series
    if not pd.api.types.is_datetime64_any_dtype(s.dtype):
        if _is_string_source(s):
            s = pd.to_datetime(s.astype(object), errors="coerce", format="mixed")
        else:
            s = pd.to_datetime(s.astype(object), errors="coerce")
        if not pd.api.types.is_datetime64_any_dtype(s.dtype):  # mixed UTC offsets
            s = pd.to_datetime(series.astype(object), errors="coerce", format="mixed",
                               utc=True)
    tz = getattr(s.dt, "tz", None)
    if tz is not None:
        s = s.dt.tz_localize(None)
    return s, (str(tz) if tz is not None else None)


def date_time_parts(ts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Days since 1970-01-01 and whole seconds from midnight, NaN where missing."""
    missing = ts.isna().to_numpy()
    ns = ts.to_numpy(dtype="datetime64[ns]").view(np.int64)
    day_ns = SECONDS_PER_DAY * 10**9
    days = np.floor_divide(ns, day_ns)
    secs = (ns - days * day_ns) // 10**9
    days, secs = days.astype(float), secs.astype(float)
    days[missing] = np.nan
    secs[missing] = np.nan
    return days, secs


def time_seconds(series: pd.Series) -> np.ndarray:
    """Seconds from midnight for time strings, ``datetime.time`` or timedelta values."""
    if pd.api.types.is_timedelta64_dtype(series.dtype):
        secs = series.dt.total_seconds().to_numpy(dtype=float, na_value=np.nan)
        return np.floor(secs)
    out = np.full(len(series), np.nan)
    for i, v in enumerate(series.to_numpy(dtype=object)):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
            continue
        if isinstance(v, _dt.time):
            out[i] = v.hour * 3600 + v.minute * 60 + v.second
        elif isinstance(v, (_dt.timedelta, pd.Timedelta)):
            out[i] = np.floor(pd.Timedelta(v).total_seconds())
        else:
            m = _TIME_PARTS_RE.match(str(v))
            if m and int(m.group(1)) < 24:
                out[i] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0)
    return out


def detect_resolution(secs: np.ndarray) -> int:
    s = secs[~np.isnan(secs)]
    for res in (3600, 60):
        if np.all(s % res == 0):
            return res
    return 1


def _date_format(values: pd.Series) -> str:
    m = _DATE_SEP_RE.match(str(values.iloc[0]))
    sep = m.group(1) if m else "-"
    return f"%Y{sep}%m{sep}%d"


def _string_format(series: pd.Series, t: ColumnType, resolution: int | None) -> str | None:
    if not _is_string_source(series):
        return None
    values = series.dropna().astype(str).iloc[:1000]
    with_seconds = bool(values.str.contains(r"\d:\d{2}:\d{2}", regex=True).any()) \
        or (resolution is not None and resolution < 60)
    time_fmt = "%H:%M:%S" if with_seconds else "%H:%M"
    if t is ColumnType.TIME:
        return time_fmt
    date_fmt = _date_format(values)
    if t is ColumnType.DATE:
        return date_fmt
    joiner = "T" if values.str.contains(r"\dT\d", regex=True).any() else " "
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


def profile_temporal(name: str, series: pd.Series, t: ColumnType, *, n_rows: int,
                     options: dict[str, Any]) -> ColumnResult:
    """Profile a date, datetime or time column (min/max and grids follow the k rule)."""
    k = options["min_tail_count"]
    warnings: list[WarningRecord] = []
    common: dict[str, Any] = dict(name=name, type=t, source_dtype=str(series.dtype))

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


def format_times(secs: np.ndarray, fmt: str | None, source_dtype: str) -> Any:
    s = secs.astype(np.int64)
    if source_dtype.startswith("timedelta"):
        return pd.to_timedelta(s, unit="s").to_numpy()
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    if fmt is None:
        return np.array([_dt.time(a, b, c) for a, b, c in zip(h, m, sec, strict=True)],
                        dtype=object)
    with_seconds = "%S" in fmt
    return np.array([f"{a:02d}:{b:02d}:{c:02d}" if with_seconds else f"{a:02d}:{b:02d}"
                     for a, b, c in zip(h, m, sec, strict=True)], dtype=object)
