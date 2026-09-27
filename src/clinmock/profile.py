"""``profile_dataframe``: build a shareable statistical profile from a DataFrame."""

from __future__ import annotations

import datetime as _dt
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .dependence import nominal_fixed_mask, observed_to_latent, pairwise_latent_correlation
from .exceptions import ClinmockWarning, PrivacyWarning
from .marginals import ColumnResult
from .marginals.categorical import profile_categorical
from .marginals.numeric import profile_numeric
from .psd import nearest_correlation
from .schema import (
    DEFAULT_OPTIONS,
    CategoricalColumn,
    ConstantColumn,
    CorrelationMatrix,
    Dependence,
    IdColumn,
    LevelMap,
    Profile,
    ReasonColumn,
    WarningRecord,
    latent_labels,
    nominal_blocks,
)
from .types import CATEGORICAL_TYPES, TEMPORAL_TYPES, ColumnType, TypeSpec, resolve_types

LARGE_PSD_CHANGE = 0.1
_PRIVACY_CODES = {"rare_levels_merged", "rare_levels_folded", "rare_binary_level",
                  "rare_ordinal_level"}


def _constant(name: str, series: pd.Series, n_rows: int) -> ColumnResult:
    values = series.dropna()
    v = values.iloc[0]
    if isinstance(v, (pd.Timestamp, _dt.datetime)):
        ts = pd.Timestamp(v)
        is_date = ts == ts.normalize()
        value, kind = (ts.date().isoformat(), "date") if is_date else (ts.isoformat(), "datetime")
    elif isinstance(v, _dt.date):
        value, kind = v.isoformat(), "date"
    elif isinstance(v, _dt.time):
        value, kind = v.isoformat(), "time"
    elif isinstance(v, (bool, np.bool_)):
        value, kind = bool(v), "bool"
    elif isinstance(v, (int, np.integer)):
        value, kind = int(v), "int"
    elif isinstance(v, (float, np.floating)) and np.isfinite(v):
        value, kind = float(v), "float"
    else:
        value, kind = str(v), "str"
    column = ConstantColumn(name=name, type=ColumnType.CONSTANT, source_dtype=str(series.dtype),
                            n_nonmissing=len(values), missing_rate=1 - len(values) / n_rows,
                            value=value, value_kind=kind)
    return ColumnResult(column)


def _reason(name: str, t: ColumnType, series: pd.Series, n_rows: int,
            reason: str) -> ColumnResult:
    n = int(series.notna().sum())
    return ColumnResult(ReasonColumn(name=name, type=t, source_dtype=str(series.dtype),
                                     n_nonmissing=n, missing_rate=1 - n / n_rows, reason=reason))


def _profile_column(name: str, series: pd.Series, spec: TypeSpec, n_rows: int,
                    options: dict[str, Any]) -> ColumnResult:
    t = spec.type
    if t in TEMPORAL_TYPES:
        result = _reason(name, ColumnType.EXCLUDED, series, n_rows,
                         f"{t.value} columns are not profiled yet (planned for M6)")
        result.warnings.append(WarningRecord(name, "temporal_not_supported",
                                             f"{t.value} column excluded: date/time support "
                                             "is not implemented yet."))
        return result
    if t is ColumnType.ID:
        n = int(series.notna().sum())
        numeric = pd.api.types.is_numeric_dtype(series.dtype) and not \
            pd.api.types.is_bool_dtype(series.dtype)
        return ColumnResult(IdColumn(name=name, type=t, source_dtype=str(series.dtype),
                                     n_nonmissing=n, missing_rate=1 - n / n_rows,
                                     value_kind="int" if numeric else "str",
                                     reason=spec.reason or "identifier"))
    if t is ColumnType.TEXT:
        return _reason(name, t, series, n_rows, spec.reason or "free text")
    if t is ColumnType.EXCLUDED:
        return _reason(name, t, series, n_rows, "excluded by the user")
    if series.notna().sum() == 0:
        return _reason(name, ColumnType.EMPTY, series, n_rows, "all values missing")
    explicit_levels = t in CATEGORICAL_TYPES and spec.levels is not None \
        and spec.source == "override"
    try:
        n_unique = series.nunique()
    except TypeError:
        n_unique = series.dropna().astype(str).nunique()
    if t is ColumnType.CONSTANT or (n_unique == 1 and not explicit_levels):
        return _constant(name, series, n_rows)
    if t is ColumnType.NUMERIC:
        return profile_numeric(name, series, n_rows=n_rows, options=options)
    return profile_categorical(name, series, spec, n_rows=n_rows, options=options)


def _matrix_to_list(m: np.ndarray) -> list[list[float | None]]:
    d = m.shape[0]
    out: list[list[float | None]] = [[None] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            v = m[i, j]
            value = None if np.isnan(v) else float(np.clip(v, -1.0, 1.0))
            out[i][j] = out[j][i] = value
    return out


def _estimate_dependence(results: list[ColumnResult], n_rows: int, options: dict[str, Any],
                         warns: list[WarningRecord]) -> Dependence:
    columns = [r.column for r in results]
    labels = latent_labels(columns)
    features = [f for r in results if r.column.has_latent for f in r.features]
    cells = [c for r in results if r.column.has_latent for c in r.feature_cells]
    if len(features) != len(labels) or len(cells) != len(labels):
        raise AssertionError("latent features do not match latent labels")
    x = np.column_stack(features) if features else np.zeros((n_rows, 0))
    blocks = nominal_blocks(labels)
    observed, n_pairs, n_sparse = pairwise_latent_correlation(
        x, cells, blocks, options["min_pair_count"])
    if n_sparse:
        warns.append(WarningRecord(None, "sparse_pairs",
                                   f"{n_sparse} variable pairs have fewer than "
                                   f"{options['min_pair_count']} jointly observed rows; their "
                                   "correlation is set to 0."))
    nominal_p = {c.name: np.array(c.probabilities) for c in columns
                 if isinstance(c, CategoricalColumn) and c.type is ColumnType.NOMINAL}
    raw_latent = observed_to_latent(np.nan_to_num(observed, nan=0.0), labels, nominal_p)
    latent, info = nearest_correlation(raw_latent, nominal_fixed_mask(labels))
    if info["applied"] and info["max_abs_change"] > LARGE_PSD_CHANGE:
        warns.append(WarningRecord(None, "large_psd_correction",
                                   f"nearest-PSD correction changed a latent correlation by "
                                   f"{info['max_abs_change']:.3f}."))
    return Dependence(
        observed=CorrelationMatrix(labels, _matrix_to_list(observed),
                                   [[int(v) for v in row] for row in n_pairs]),
        latent=CorrelationMatrix(labels, _matrix_to_list(latent)),
        psd_correction=info,
    )


def profile_dataframe(
    df: pd.DataFrame,
    types: Mapping[str, Any] | None = None,
    *,
    pseudonymize: Sequence[str] | None = None,
    rare_threshold: int = DEFAULT_OPTIONS["rare_threshold"],
    min_tail_count: int = DEFAULT_OPTIONS["min_tail_count"],
    quantile_grid: int = DEFAULT_OPTIONS["quantile_grid"],
    discrete_max_levels: int = DEFAULT_OPTIONS["discrete_max_levels"],
    max_levels: int = DEFAULT_OPTIONS["max_levels"],
    ordinal_max_levels: int = DEFAULT_OPTIONS["ordinal_max_levels"],
    min_pair_count: int = DEFAULT_OPTIONS["min_pair_count"],
) -> Profile:
    """Profile ``df`` into aggregate statistics (no row-level data).

    Level names are stored as-is except for columns listed in ``pseudonymize`` (or with
    ``"pseudonymize": True`` in ``types``); their pseudonym map is kept only in memory as
    ``profile.level_map`` and must be saved separately with ``save_level_map``.
    """
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("df has no rows")
    options = dict(DEFAULT_OPTIONS)
    options.update(rare_threshold=rare_threshold, min_tail_count=min_tail_count,
                   quantile_grid=quantile_grid, discrete_max_levels=discrete_max_levels,
                   max_levels=max_levels, ordinal_max_levels=ordinal_max_levels,
                   min_pair_count=min_pair_count)
    for key in ("rare_threshold", "min_tail_count", "quantile_grid", "min_pair_count"):
        if options[key] < (2 if key == "quantile_grid" else 1):
            raise ValueError(f"{key} is too small")

    resolution = resolve_types(df, types, pseudonymize=pseudonymize, max_levels=max_levels,
                               ordinal_max_levels=ordinal_max_levels)
    resolution.emit_warnings(stacklevel=3)
    records = [WarningRecord(w.column, w.code, w.message) for w in resolution.warnings]

    results: list[ColumnResult] = []
    level_map: dict[str, dict] = {}
    for name in df.columns:
        result = _profile_column(name, df[name], resolution.specs[name], n_rows, options)
        for w in result.warnings:
            category = PrivacyWarning if w.code in _PRIVACY_CODES else ClinmockWarning
            warnings.warn(f"[{w.column}] {w.message}", category, stacklevel=2)
        records.extend(result.warnings)
        if result.level_map:
            level_map[name] = result.level_map
        results.append(result)

    dependence_warnings: list[WarningRecord] = []
    dependence = _estimate_dependence(results, n_rows, options, dependence_warnings)
    for w in dependence_warnings:
        warnings.warn(w.message, ClinmockWarning, stacklevel=2)
    records.extend(dependence_warnings)

    return Profile(n_rows=n_rows, columns=[r.column for r in results], options=options,
                   dependence=dependence, warnings=records,
                   level_map=LevelMap(level_map) if level_map else None)
