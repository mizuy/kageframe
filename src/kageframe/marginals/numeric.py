from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..latent import interval_scores
from ..schema import NumericColumn, Quantiles, WarningRecord
from ..types import ColumnType
from . import ColumnResult

SUMMARY_PROBS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
TAIL_PROBS = (0.001, 0.002, 0.005, 0.995, 0.998, 0.999)
MAX_DECIMALS = 6


def detect_decimals(x: np.ndarray) -> int | None:
    """Smallest number of decimals that reproduces every value; None if more than 6."""
    scale = np.maximum(1.0, np.abs(x))
    for d in range(MAX_DECIMALS + 1):
        if np.all(np.abs(np.round(x, d) - x) <= 1e-9 * scale):
            return d
    return None


def round_value(v: float, decimals: int | None) -> float:
    return float(v) if decimals is None else float(np.round(v, decimals))


def to_float_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        series = series.astype("float64")
    elif not pd.api.types.is_numeric_dtype(series.dtype):
        series = pd.to_numeric(series, errors="coerce")
    return series.astype("float64").to_numpy(dtype=float, na_value=np.nan)


def profile_numeric(name: str, series: pd.Series, *, n_rows: int,
                    options: dict[str, Any]) -> ColumnResult:
    x_all = to_float_array(series)
    observed = ~np.isnan(x_all)
    x = x_all[observed]
    n = len(x)
    k = options["min_tail_count"]
    integer = bool(np.all(x == np.round(x)))
    decimals = 0 if integer else detect_decimals(x)
    warnings: list[WarningRecord] = []

    uniq, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    cells = counts / n
    scores, _ = interval_scores(cells)
    feature = np.full(n_rows, np.nan)
    feature[observed] = scores[inverse]

    common: dict[str, Any] = dict(
        name=name, type=ColumnType.NUMERIC, source_dtype=str(series.dtype), n_nonmissing=n,
        missing_rate=1 - n / n_rows, subtype="integer" if integer else "float",
        decimals=decimals, mean=float(np.mean(x)),
        sd=float(np.std(x, ddof=1)) if n > 1 else None,
    )

    if n < 2 * k:
        warnings.append(WarningRecord(name, "too_few_values",
                                      f"fewer than {2 * k} non-missing values; no distribution "
                                      "is stored and the column is generated as missing."))
        column = NumericColumn(**common, min=None, max=None, summary_quantiles=None,
                               mode="quantile", grid=None)
        return ColumnResult(column, [feature], [cells], warnings)

    lo, hi = k / n, 1 - k / n

    def quantiles(probs: np.ndarray) -> list[float]:
        return [float(v) for v in np.quantile(x, np.clip(probs, lo, hi))]

    summary = Quantiles(list(SUMMARY_PROBS), quantiles(np.array(SUMMARY_PROBS)))

    discrete = (integer and len(uniq) <= options["discrete_max_levels"]
                and int(counts.min()) >= k)
    if discrete:
        column = NumericColumn(**common, min=float(uniq[0]), max=float(uniq[-1]),
                               summary_quantiles=summary, mode="discrete",
                               support=[float(v) for v in uniq],
                               probabilities=[float(c / n) for c in counts])
    else:
        n_grid = int(min(options["quantile_grid"], max(2, n // k)))
        probs = np.linspace(0.0, 1.0, n_grid)
        # Extra tail points keep linear interpolation from inflating heavy tails; each is
        # still backed by at least k rows.
        tail = np.array([p for p in TAIL_PROBS if lo <= p <= hi])
        probs = np.unique(np.concatenate([probs, tail]))
        values = quantiles(probs)
        column = NumericColumn(**common, min=round_value(values[0], decimals),
                               max=round_value(values[-1], decimals),
                               summary_quantiles=summary, mode="quantile",
                               grid=Quantiles([float(p) for p in probs], values))
    return ColumnResult(column, [feature], [cells], warnings)
