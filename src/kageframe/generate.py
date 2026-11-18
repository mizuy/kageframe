"""``generate``: sample a dummy DataFrame from a profile."""

from __future__ import annotations

import datetime as _dt
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import special

from ._frame import parse_dtype
from .dependence import nominal_fixed_mask
from .exceptions import KageFrameWarning
from .latent import nominal_choice, thresholds
from .marginals.temporal import (
    build_dates,
    clip_seconds,
    datetime_values,
    format_times,
    sample_part,
)
from .psd import EIG_FLOOR, nearest_correlation
from .schema import (
    CategoricalColumn,
    ColumnProfile,
    ConstantColumn,
    DateColumn,
    DatetimeColumn,
    IdColumn,
    LevelMap,
    NumericColumn,
    Profile,
    TimeColumn,
)
from .types import ColumnType

# Stream indices of SeedSequence(seed).spawn(); the missingness stream is split per column.
_LATENT_STREAM = 0
_MISSING_STREAM = 1
_N_STREAMS = 2
MISSING_MODES = ("exact", "bernoulli", "none")


def _latent_factor(profile: Profile) -> np.ndarray:
    labels = profile.latent_labels()
    if not labels:
        return np.zeros((0, 0))
    r = profile.dependence.latent_matrix()
    try:
        return np.linalg.cholesky(r)
    except np.linalg.LinAlgError:
        if not profile.dependence.psd_correction.get("applied"):
            warnings.warn("latent correlation matrix is not positive definite; applying the "
                          "nearest-PSD correction before sampling", KageFrameWarning,
                          stacklevel=3)
        fixed, _ = nearest_correlation(r, nominal_fixed_mask(labels), eps=EIG_FLOOR)
        return np.linalg.cholesky(fixed)


def _latent_width(col: ColumnProfile) -> int:
    if not col.has_latent:
        return 0
    if isinstance(col, CategoricalColumn) and col.type is ColumnType.NOMINAL:
        return len(col.levels)
    if col.type is ColumnType.DATETIME:
        return 2
    return 1


# ---------------------------------------------------------------------------
# per-type reconstruction
# ---------------------------------------------------------------------------


def _numeric(col: NumericColumn, z: np.ndarray) -> np.ndarray:
    if col.mode == "discrete":
        idx = np.searchsorted(thresholds(np.array(col.probabilities)), z, side="right")
        x = np.array(col.support)[idx]
    elif col.grid is None:
        return np.full(len(z), np.nan)
    else:
        x = np.interp(special.ndtr(z), col.grid.probs, col.grid.values)
        if col.decimals is not None:
            x = np.round(x, col.decimals)
    if col.subtype == "integer":
        return np.round(x).astype(np.int64)
    return x.astype(float)


def _level_array(levels: list, kind: str) -> np.ndarray:
    if kind == "bool" and all(isinstance(v, bool) for v in levels):
        return np.array(levels, dtype=bool)
    if kind == "int" and all(isinstance(v, int) for v in levels):
        return np.array(levels, dtype=np.int64)
    if kind == "float" and all(isinstance(v, (int, float)) for v in levels):
        return np.array(levels, dtype=float)
    return np.array(levels, dtype=object)


def _mapped_levels(col: CategoricalColumn, level_map: LevelMap | None) -> list:
    levels = list(col.levels)
    if level_map is not None and col.name in level_map.columns:
        mapping = level_map.columns[col.name]
        levels = [mapping.get(str(lv), lv) for lv in levels]
    return levels


def _categorical(col: CategoricalColumn, z: np.ndarray, level_map: LevelMap | None
                 ) -> np.ndarray:
    p = np.array(col.probabilities)
    if col.type is ColumnType.NOMINAL:
        idx = nominal_choice(z, p)
    else:
        idx = np.searchsorted(thresholds(p), z[:, 0], side="right")
    return _level_array(_mapped_levels(col, level_map), col.value_kind)[idx]


def _date(col: DateColumn, z: np.ndarray) -> np.ndarray:
    days = sample_part(col.date_part, special.ndtr(z[:, 0]), 1)
    if days is None:
        return np.full(len(z), np.datetime64("NaT", "D"))
    return build_dates(days)


def _datetime(col: DatetimeColumn, z: np.ndarray) -> np.ndarray:
    res = col.time_part.resolution_seconds or 1
    days = sample_part(col.date_part, special.ndtr(z[:, 0]), 1)
    secs = sample_part(col.time_part, special.ndtr(z[:, 1]), res)
    if days is None or secs is None:
        return np.full(len(z), np.datetime64("NaT", "s"))
    return datetime_values(days, clip_seconds(secs, res))


def _time_seconds(col: TimeColumn, z: np.ndarray) -> np.ndarray | None:
    res = col.time_part.resolution_seconds or 1
    secs = sample_part(col.time_part, special.ndtr(z[:, 0]), res)
    return None if secs is None else clip_seconds(secs, res)


def _constant(col: ConstantColumn, n: int, level_map: LevelMap | None) -> pl.Series:
    value = col.value
    if level_map is not None and col.name in level_map.columns:
        value = level_map.columns[col.name].get(str(value), value)
    kind = col.value_kind
    if kind == "date":
        return pl.Series([_dt.date.fromisoformat(str(value))] * n, dtype=pl.Date)
    if kind == "datetime":
        v = _dt.datetime.fromisoformat(str(value))
        if v.tzinfo is not None:
            v = v.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return pl.Series([v] * n, dtype=pl.Datetime("us"))
    if kind == "time":
        return pl.Series([_dt.time.fromisoformat(str(value))] * n, dtype=pl.Time)
    dtype = {"bool": pl.Boolean, "int": pl.Int64, "float": pl.Float64}.get(kind, pl.String)
    return pl.Series([value] * n, dtype=dtype)


def _ids(col: IdColumn, n: int) -> pl.Series:
    if col.value_kind == "int":
        return pl.Series(np.arange(1, n + 1, dtype=np.int64))
    width = max(col.width, len(str(n)))
    return pl.Series([f"{col.prefix}{i:0{width}d}" for i in range(1, n + 1)], dtype=pl.String)


# ---------------------------------------------------------------------------
# missingness and dtypes
# ---------------------------------------------------------------------------


def missing_mask(n: int, rate: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    """MCAR mask: ``exact`` hides ``round(n * rate)`` rows, ``bernoulli`` each row w.p. rate."""
    mask = np.zeros(n, dtype=bool)
    if mode == "none" or rate <= 0:
        return mask
    if mode == "bernoulli":
        return rng.random(n) < rate
    k = int(round(n * rate))
    if k:
        mask[rng.choice(n, size=k, replace=False)] = True
    return mask


def _castable(have: pl.DataType, want: pl.DataType) -> bool:
    if have.is_numeric() and want.is_numeric():
        return True
    temporal = (pl.Date, pl.Datetime)
    return have.base_type() in temporal and want.base_type() in temporal


def _restore_dtype(col: ColumnProfile, s: pl.Series, levels: list | None) -> pl.Series:
    """Cast to a dtype close to the source column (``col.source_dtype``)."""
    src = col.source_dtype
    if levels is not None and s.dtype == pl.String:
        if src == "Enum":
            return s.cast(pl.Enum([str(v) for v in levels]))
        return s.cast(pl.Categorical())
    want = parse_dtype(src)
    if want is not None and s.dtype != want and _castable(s.dtype, want):
        s = s.cast(want)
    tz = getattr(col, "tz", None)
    if tz and s.dtype.base_type() is pl.Datetime:
        s = s.dt.replace_time_zone(tz, ambiguous="latest", non_existent="null")
    return s


def _finalize(col: ColumnProfile, values: Any, mask: np.ndarray,
              levels: list | None = None) -> pl.Series:
    """Apply the missing mask and restore a dtype close to the source column."""
    if isinstance(values, np.ndarray) and values.dtype.kind == "M" \
            and values.dtype != np.dtype("datetime64[D]"):
        values = values.astype("datetime64[us]")
    s = values if isinstance(values, pl.Series) else pl.Series(values)
    if s.dtype == pl.Object:
        s = pl.Series([None if v is None else str(v) for v in s.to_list()], dtype=pl.String)
    if s.dtype.is_float():
        s = s.fill_nan(None)
    s = _restore_dtype(col, s, levels)
    if mask.any():
        s = s.scatter(np.flatnonzero(mask), None)
    return s.alias(col.name)


def _resolve_level_map(level_map: LevelMap | dict | str | Path | None) -> LevelMap | None:
    if level_map is None or isinstance(level_map, LevelMap):
        return level_map
    if isinstance(level_map, dict):
        return LevelMap.from_dict(level_map)
    return LevelMap.load(level_map)


def generate(
    profile: Profile,
    n: int | None = None,
    seed: int | None = None,
    *,
    missing: str = "exact",
    level_map: LevelMap | dict | str | Path | None = None,
    id_values: Mapping[str, Sequence[Any] | pl.Series] | None = None,
) -> pl.DataFrame:
    """Generate a dummy DataFrame from ``profile``.

    Parameters
    ----------
    profile:
        A profile from :func:`profile_dataframe` or :func:`load_profile`.
    n:
        Number of rows (default: the number of rows that was profiled).
    seed:
        Seed for the random generator. The same profile, ``n``, ``seed`` and ``missing``
        give identical output with the same numpy version.
    missing:
        How to reproduce each column's missing rate (missing completely at random):
        ``"exact"`` (default) hides exactly ``round(n * missing_rate)`` rows,
        ``"bernoulli"`` hides each row independently, ``"none"`` produces no missing values.
    level_map:
        A local :class:`LevelMap` (or its dict / file path) to turn pseudonymized levels
        back into the real names. Without it, pseudonyms such as ``L01`` are kept.
    id_values:
        Values to use for identifier columns instead of new sequential ids, as
        ``{column: values}`` with ``n`` values each (e.g. keys shared with another table).
        No missing values are added to these columns.

    Returns
    -------
    polars.DataFrame
        One column per profiled column, in the original order, except free-text and
        excluded columns. Identifier columns get new sequential values unless given in
        ``id_values``.
    """
    n = profile.n_rows if n is None else int(n)
    if n < 1:
        raise ValueError("n must be positive")
    if missing not in MISSING_MODES:
        raise ValueError(f"missing must be one of {MISSING_MODES}")
    lm = _resolve_level_map(level_map)
    ids = _check_id_values(profile, id_values, n)
    streams = np.random.SeedSequence(seed).spawn(_N_STREAMS)
    rng = np.random.default_rng(streams[_LATENT_STREAM])
    missing_streams = streams[_MISSING_STREAM].spawn(len(profile.columns))

    factor = _latent_factor(profile)
    d = factor.shape[0]
    z = rng.standard_normal((n, d)) @ factor.T if d else np.zeros((n, 0))

    offset = 0
    data: list[pl.Series] = []
    for i, col in enumerate(profile.columns):
        width = _latent_width(col)
        block = z[:, offset:offset + width]
        offset += width
        if not col.in_output:
            continue
        values: Any
        categories = None
        if isinstance(col, NumericColumn):
            values = _numeric(col, block[:, 0])
        elif isinstance(col, CategoricalColumn):
            values = _categorical(col, block, lm)
            if col.source_dtype in ("Categorical", "Enum"):
                categories = _mapped_levels(col, lm)
        elif isinstance(col, DateColumn):
            values = _date(col, block)
            if col.output_format:
                values = _as_strings(values, col.output_format)
        elif isinstance(col, DatetimeColumn):
            values = _datetime(col, block)
            if col.output_format:
                values = _as_strings(values, col.output_format)
        elif isinstance(col, TimeColumn):
            secs = _time_seconds(col, block)
            values = (pl.Series([None] * n, dtype=pl.String) if secs is None
                      else format_times(secs, col.output_format, col.source_dtype))
        elif isinstance(col, IdColumn):
            if col.name in ids:
                data.append(ids[col.name])
                continue
            values = _ids(col, n)
        elif isinstance(col, ConstantColumn):
            values = _constant(col, n, lm)
        elif col.type is ColumnType.EMPTY:
            data.append(pl.Series(col.name, [None] * n,
                                  dtype=parse_dtype(col.source_dtype) or pl.Null))
            continue
        else:
            raise NotImplementedError(f"generation of {col.type.value} columns is not "
                                      "supported")
        mask = missing_mask(n, col.missing_rate, missing,
                            np.random.default_rng(missing_streams[i]))
        data.append(_finalize(col, values, mask, categories))
    return pl.DataFrame(data)


def _as_strings(values: np.ndarray, fmt: str) -> pl.Series:
    if values.dtype != np.dtype("datetime64[D]"):
        values = values.astype("datetime64[us]")
    return pl.Series(values).dt.to_string(fmt)


def _check_id_values(profile: Profile, id_values: Mapping[str, Any] | None,
                     n: int) -> dict[str, pl.Series]:
    if not id_values:
        return {}
    id_columns = {c.name for c in profile.columns if isinstance(c, IdColumn)}
    unknown = sorted(set(id_values) - id_columns)
    if unknown:
        raise ValueError(f"id_values refers to columns that are not id columns: {unknown}")
    out = {}
    for name, values in id_values.items():
        s = values if isinstance(values, pl.Series) else pl.Series(values)
        if len(s) != n:
            raise ValueError(f"id_values[{name!r}] has {len(s)} values; expected n={n}")
        out[name] = s.alias(name)
    return out
