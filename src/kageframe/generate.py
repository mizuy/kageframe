"""``generate``: sample a dummy DataFrame from a profile."""

from __future__ import annotations

import datetime as _dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special

from .dependence import nominal_fixed_mask
from .exceptions import KageFrameWarning
from .latent import nominal_choice, thresholds
from .psd import EIG_FLOOR, nearest_correlation
from .schema import (
    CategoricalColumn,
    ColumnProfile,
    ConstantColumn,
    IdColumn,
    LevelMap,
    NumericColumn,
    Profile,
)
from .types import ColumnType

# Stream indices of SeedSequence(seed).spawn(); index 1 is reserved for missingness (M5).
_LATENT_STREAM = 0
_N_STREAMS = 2


def _latent_factor(profile: Profile) -> np.ndarray:
    labels = profile.latent_labels()
    if not labels:
        return np.zeros((0, 0))
    r = np.array(profile.dependence.latent.matrix, dtype=float)
    try:
        return np.linalg.cholesky(r)
    except np.linalg.LinAlgError:
        warnings.warn("latent correlation matrix is not positive definite; applying the "
                      "nearest-PSD correction before sampling", KageFrameWarning, stacklevel=3)
        fixed, _ = nearest_correlation(r, nominal_fixed_mask(labels), eps=EIG_FLOOR)
        return np.linalg.cholesky(fixed)


def _round(x: np.ndarray, decimals: int | None) -> np.ndarray:
    return x if decimals is None else np.round(x, decimals)


def _numeric(col: NumericColumn, z: np.ndarray) -> np.ndarray:
    if col.mode == "discrete":
        idx = np.searchsorted(thresholds(np.array(col.probabilities)), z, side="right")
        x = np.array(col.support)[idx]
    elif col.grid is None:
        return np.full(len(z), np.nan)
    else:
        x = np.interp(special.ndtr(z), col.grid.probs, col.grid.values)
        x = _round(x, col.decimals)
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


def _categorical(col: CategoricalColumn, z: np.ndarray, level_map: LevelMap | None
                 ) -> np.ndarray:
    levels = list(col.levels)
    if level_map is not None and col.name in level_map.columns:
        mapping = level_map.columns[col.name]
        levels = [mapping.get(str(lv), lv) for lv in levels]
    p = np.array(col.probabilities)
    if col.type is ColumnType.NOMINAL:
        idx = nominal_choice(z, p)
    else:
        idx = np.searchsorted(thresholds(p), z[:, 0], side="right")
    return _level_array(levels, col.value_kind)[idx]


def _constant(col: ConstantColumn, n: int, level_map: LevelMap | None) -> np.ndarray | pd.Series:
    value = col.value
    if level_map is not None and col.name in level_map.columns:
        value = level_map.columns[col.name].get(str(value), value)
    kind = col.value_kind
    if kind in ("date", "datetime"):
        return pd.Series(pd.Timestamp(value), index=range(n))
    if kind == "time":
        return np.full(n, _dt.time.fromisoformat(str(value)), dtype=object)
    dtype = {"bool": bool, "int": np.int64, "float": float}.get(kind, object)
    return np.full(n, value, dtype=dtype)


def _ids(col: IdColumn, n: int) -> np.ndarray:
    if col.value_kind == "int":
        return np.arange(1, n + 1, dtype=np.int64)
    width = max(col.width, len(str(n)))
    return np.array([f"{col.prefix}{i:0{width}d}" for i in range(1, n + 1)], dtype=object)


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
    level_map: LevelMap | dict | str | Path | None = None,
) -> pd.DataFrame:
    """Generate ``n`` rows (default: ``profile.n_rows``) from ``profile``.

    The same profile, ``n`` and ``seed`` give identical output in the same environment.
    Pseudonymized levels stay pseudonyms unless a local ``level_map`` is given.
    Free-text and excluded columns are not part of the output.
    """
    n = profile.n_rows if n is None else int(n)
    if n < 1:
        raise ValueError("n must be positive")
    lm = _resolve_level_map(level_map)
    streams = np.random.SeedSequence(seed).spawn(_N_STREAMS)
    rng = np.random.default_rng(streams[_LATENT_STREAM])

    factor = _latent_factor(profile)
    d = factor.shape[0]
    z = rng.standard_normal((n, d)) @ factor.T if d else np.zeros((n, 0))

    offset = 0
    data: dict[str, object] = {}
    for col in profile.columns:
        width = _latent_width(col)
        block = z[:, offset:offset + width]
        offset += width
        if not col.in_output:
            continue
        if isinstance(col, NumericColumn):
            data[col.name] = _numeric(col, block[:, 0])
        elif isinstance(col, CategoricalColumn):
            data[col.name] = _categorical(col, block, lm)
        elif isinstance(col, IdColumn):
            data[col.name] = _ids(col, n)
        elif isinstance(col, ConstantColumn):
            data[col.name] = _constant(col, n, lm)
        elif col.type is ColumnType.EMPTY:
            data[col.name] = np.full(n, np.nan)
        else:
            raise NotImplementedError(f"generation of {col.type.value} columns is not "
                                      "implemented yet")
    return pd.DataFrame(data, index=pd.RangeIndex(n))


def _latent_width(col: ColumnProfile) -> int:
    if not col.has_latent:
        return 0
    if isinstance(col, CategoricalColumn) and col.type is ColumnType.NOMINAL:
        return len(col.levels)
    if col.type is ColumnType.DATETIME:
        return 2
    return 1
