from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import special

from ..latent import interval_scores, normal_pdf, thresholds
from ..schema import CategoricalColumn, ConstantColumn, Scalar, WarningRecord
from ..types import ColumnType, TypeSpec, has_string_levels
from . import ColumnResult

OTHER = "Other"
OTHER_FALLBACK = "__other__"


def value_kind(series: pd.Series) -> str:
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        dtype = dtype.categories.dtype
    values = series.dropna()
    if pd.api.types.is_bool_dtype(dtype) or (
        len(values) and all(isinstance(v, (bool, np.bool_)) for v in values.iloc[:1000])
    ):
        return "bool"
    if not has_string_levels(series) and pd.api.types.is_numeric_dtype(dtype):
        arr = values.astype("float64").to_numpy()
        return "int" if np.all(arr == np.round(arr)) else "float"
    return "str"


def cast_scalar(v: Any, kind: str) -> Scalar:
    if kind == "bool":
        return bool(v)
    if kind == "int":
        return int(round(float(v)))
    if kind == "float":
        return float(v)
    return str(v)


def normalize_values(values: pd.Series, kind: str) -> np.ndarray:
    if kind == "bool":
        return values.astype(bool).to_numpy()
    if kind in ("int", "float"):
        arr = values.astype("float64").to_numpy()
        return np.round(arr).astype(np.int64) if kind == "int" else arr
    return values.astype(object).map(str).to_numpy(dtype=object)


def _sort_key(v: Scalar) -> tuple[int, Any]:
    return (0, v) if not isinstance(v, str) else (1, v)


def _merge_rare_nominal(name: str, levels: list[Scalar], counts: np.ndarray, k: int
                        ) -> tuple[list[Scalar], np.ndarray, str | None, int,
                                   list[WarningRecord]]:
    """Return new levels, an old->new index map, the other level, #merged, warnings."""
    rare = [i for i, c in enumerate(counts) if 0 < c < k]
    if not rare:
        return levels, np.arange(len(levels)), None, 0, []
    existing = next((i for i, lv in enumerate(levels) if lv == OTHER), None)
    merged = [i for i in rare if i != existing]
    if not merged:
        return levels, np.arange(len(levels)), None, 0, []
    pooled = int(counts[merged].sum()) + (int(counts[existing]) if existing is not None else 0)
    keep = [i for i in range(len(levels)) if i not in merged and i != existing]
    mapping = np.empty(len(levels), dtype=np.int64)
    for new, old in enumerate(keep):
        mapping[old] = new
    warnings: list[WarningRecord] = []
    if pooled >= k:
        other = OTHER
        if existing is None and any(str(lv).lower() == "other" for lv in levels):
            other = OTHER_FALLBACK
        new_levels = [levels[i] for i in keep] + [other]
        mapping[merged] = len(keep)
        if existing is not None:
            mapping[existing] = len(keep)
        warnings.append(WarningRecord(name, "rare_levels_merged",
                                      f"{len(merged)} levels with fewer than {k} rows were "
                                      f"merged into {other!r}."))
        return new_levels, mapping, other, len(merged), warnings
    # Too few rows even when pooled: fold them into the most frequent remaining level.
    if existing is not None:
        keep.append(existing)
        mapping[existing] = len(keep) - 1
    target = max(range(len(keep)), key=lambda j: counts[keep[j]])
    mapping[merged] = target
    warnings.append(WarningRecord(name, "rare_levels_folded",
                                  f"{len(merged)} levels with fewer than {k} rows in total were "
                                  "folded into the most frequent level."))
    return [levels[i] for i in keep], mapping, None, len(merged), warnings


def _pseudonymize(levels: list[Scalar], counts: np.ndarray, ordered: bool,
                  other: str | None) -> tuple[list[Scalar], dict[str, Scalar]]:
    targets = [i for i, lv in enumerate(levels) if lv != other]
    if not ordered:
        targets.sort(key=lambda i: (-counts[i], _sort_key(levels[i])))
    width = max(2, len(str(len(targets))))
    new_levels = list(levels)
    level_map: dict[str, Scalar] = {}
    for rank, i in enumerate(targets, start=1):
        pseudo = f"L{rank:0{width}d}"
        level_map[pseudo] = levels[i]
        new_levels[i] = pseudo
    return new_levels, level_map


def profile_categorical(name: str, series: pd.Series, spec: TypeSpec, *, n_rows: int,
                        options: dict[str, Any]) -> ColumnResult:
    """Profile a binary/ordinal/nominal column (constant if it collapses to one level)."""
    t = spec.type
    observed_mask = series.notna().to_numpy()
    kind = value_kind(series)
    values = normalize_values(series[observed_mask], kind)
    n = len(values)
    k = options["rare_threshold"]
    warnings: list[WarningRecord] = []

    if spec.levels is not None:
        levels: list[Scalar] = [cast_scalar(v, kind) for v in spec.levels]
    else:
        levels = sorted({cast_scalar(v, kind) for v in values}, key=_sort_key)
    codes = pd.Categorical(values, categories=levels).codes.astype(np.int64)
    counts = np.bincount(codes, minlength=len(levels))

    other, n_merged = None, 0
    if t is ColumnType.NOMINAL:
        if spec.levels is None:
            order = sorted(range(len(levels)), key=lambda i: (-counts[i], _sort_key(levels[i])))
            levels = [levels[i] for i in order]
            remap = np.empty(len(order), dtype=np.int64)
            remap[order] = np.arange(len(order))
            codes, counts = remap[codes], counts[order]
        levels, mapping, other, n_merged, merge_warns = _merge_rare_nominal(name, levels,
                                                                            counts, k)
        warnings.extend(merge_warns)
        codes = mapping[codes]
        counts = np.bincount(codes, minlength=len(levels))
    elif 0 < counts[counts > 0].min() < k:
        warnings.append(WarningRecord(name, f"rare_{t.value}_level",
                                      f"a level has fewer than {k} rows; its count is stored "
                                      "as-is."))
    level_map = None
    if len(levels) < 2:
        value = levels[0]
        if spec.pseudonymize and kind == "str" and value != other:
            level_map = {"L01": value}
            value = "L01"
        warnings.append(WarningRecord(name, "single_level",
                                      "only one level remains; stored as a constant column."))
        column = ConstantColumn(name=name, type=ColumnType.CONSTANT,
                                source_dtype=str(series.dtype), n_nonmissing=n,
                                missing_rate=1 - n / n_rows, value=value, value_kind=kind)
        return ColumnResult(column, warnings=warnings, level_map=level_map)

    if spec.pseudonymize and kind == "str":
        levels, level_map = _pseudonymize(levels, counts, t is ColumnType.ORDINAL, other)

    probs = counts / n
    column = CategoricalColumn(
        name=name, type=t, source_dtype=str(series.dtype), n_nonmissing=n,
        missing_rate=1 - n / n_rows, levels=levels, counts=[int(c) for c in counts],
        probabilities=[float(p) for p in probs], value_kind=kind,
        pseudonymized=level_map is not None,
        thresholds=None if t is ColumnType.NOMINAL else [float(v) for v in thresholds(probs)],
        other_level=other, n_merged_levels=n_merged,
    )

    features: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    if t is ColumnType.NOMINAL:
        # One binary indicator per level; its latent cells are (not level, level).
        for j, p in enumerate(probs):
            feature = np.full(n_rows, np.nan)
            if 0 < p < 1:
                phi = float(normal_pdf(special.ndtri(1 - p)))
                feature[observed_mask] = np.where(codes == j, phi / p, -phi / (1 - p))
            else:
                feature[observed_mask] = 0.0
            features.append(feature)
            cells.append(np.array([1 - p, p]))
    else:
        scores, _ = interval_scores(probs)
        feature = np.full(n_rows, np.nan)
        feature[observed_mask] = scores[codes]
        features.append(feature)
        cells.append(np.asarray(probs, dtype=float))
    return ColumnResult(column, features, cells, warnings, level_map)
