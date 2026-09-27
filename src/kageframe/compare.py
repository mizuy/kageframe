"""``compare``: check how closely a dummy DataFrame follows a profile."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .latent import interval_scores
from .marginals.categorical import normalize_values, value_kind
from .marginals.numeric import to_float_array
from .marginals.temporal import date_time_parts, time_seconds, to_timestamps
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
    Quantiles,
    TimeColumn,
)
from .types import CATEGORICAL_TYPES, ColumnType

#: Default tolerances (§7 of the plan). Marginal tolerances are calibrated for 10,000
#: non-missing dummy rows and widen by ``sqrt(10000 / n)`` for smaller dummies. A
#: correlation passes if its difference is within the ``corr_<kind>`` value or within
#: ``corr_z`` standard errors of the estimator (rare binaries and ties inflate the SE).
DEFAULT_TOLERANCES: dict[str, float] = {
    "mean": 0.05,             # |mean_dummy - mean_real| / sd_real
    "sd": 0.05,               # |sd_dummy / sd_real - 1| (columns without heavy tails)
    "iqr": 0.10,              # |iqr_dummy / iqr_real - 1| (heavy-tailed columns)
    "grid_ks": 0.02,          # max CDF distance on the inner points of the profile grid
    "max_abs_dp": 0.015,      # largest per-level frequency difference
    "tvd": 0.03,              # total variation distance of level frequencies
    "missing_sigma": 3.0,     # missing rate: this many binomial standard errors ...
    "missing_abs": 0.001,     # ... plus this absolute slack
    "corr_continuous": 0.03,  # pairs of numeric / date / time dimensions
    "corr_discrete": 0.05,    # pairs involving binary / ordinal columns
    "corr_nominal": 0.07,     # pairs involving a nominal level indicator
    "corr_rmse": 0.02,        # root mean square over all compared pairs
    "corr_z": 4.0,            # per pair, also allow this many estimator standard errors
    "corr_rmse_se": 1.5,      # RMSE, also allow this multiple of the RMS standard error
}
REFERENCE_N = 10_000
#: ``(q99 - q01) / (q75 - q25)``; about 3.4 for a normal distribution.
HEAVY_TAIL_RATIO = 6.0

_CONTINUOUS = {ColumnType.NUMERIC, ColumnType.DATE, ColumnType.DATETIME, ColumnType.TIME}


@dataclass
class CheckResult:
    """Outcome of :meth:`CompareReport.check`; truthy when every check passed."""

    passed: bool
    failures: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed

    def __str__(self) -> str:
        if self.passed:
            return "all checks passed"
        return "\n".join(["checks failed:"] + [f"  - {f}" for f in self.failures])


@dataclass
class CompareReport:
    """Differences between a profile and a dummy DataFrame.

    ``marginals`` has one record per output column and ``correlations`` one record per
    pair of latent dimensions. Use :meth:`summary` / :meth:`correlation_table` for
    DataFrames, :meth:`to_dict` for JSON and :meth:`check` for a pass/fail verdict.
    """

    n_rows: int
    expected_columns: list[str]
    actual_columns: list[str]
    marginals: list[dict[str, Any]]
    correlations: list[dict[str, Any]]

    @property
    def missing_columns(self) -> list[str]:
        return [c for c in self.expected_columns if c not in self.actual_columns]

    @property
    def extra_columns(self) -> list[str]:
        return [c for c in self.actual_columns if c not in self.expected_columns]

    @property
    def columns_match(self) -> bool:
        return self.actual_columns == self.expected_columns

    def summary(self) -> pd.DataFrame:
        """One row per column: missing rate, marginal distances and worst correlation."""
        df = pd.DataFrame(self.marginals).set_index("column")
        worst: dict[str, float] = {}
        for r in self.correlations:
            for c in (r["column_a"], r["column_b"]):
                worst[c] = max(worst.get(c, 0.0), r["abs_diff"])
        df["max_corr_diff"] = pd.Series(worst)
        return df

    def correlation_table(self) -> pd.DataFrame:
        """One row per latent pair: profile vs dummy observed and generation-space values."""
        return pd.DataFrame(self.correlations)

    def correlation_summary(self) -> dict[str, float | None]:
        """Largest absolute and RMS differences of the latent correlations."""
        if not self.correlations:
            return {"max_abs_diff": None, "rmse": None, "latent_max_abs_diff": None}
        diffs = np.array([r["abs_diff"] for r in self.correlations])
        latent = [r["latent_abs_diff"] for r in self.correlations
                  if r["latent_abs_diff"] is not None]
        return {"max_abs_diff": float(diffs.max()),
                "rmse": float(np.sqrt(np.mean(diffs**2))),
                "latent_max_abs_diff": float(max(latent)) if latent else None}

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "columns": {"expected": self.expected_columns, "actual": self.actual_columns,
                        "missing": self.missing_columns, "extra": self.extra_columns,
                        "match": self.columns_match},
            "marginals": [_jsonable(r) for r in self.marginals],
            "correlations": [_jsonable(r) for r in self.correlations],
            "correlation_summary": self.correlation_summary(),
        }

    def check(self, tol: Mapping[str, float] | None = None) -> CheckResult:
        """Compare every metric with :data:`DEFAULT_TOLERANCES` (updated with ``tol``)."""
        t = dict(DEFAULT_TOLERANCES)
        if tol:
            unknown = set(tol) - set(t)
            if unknown:
                raise ValueError(f"unknown tolerances {sorted(unknown)}")
            t.update(tol)
        fails: list[str] = []
        if not self.columns_match:
            fails.append(f"columns differ: missing {self.missing_columns}, extra "
                         f"{self.extra_columns}" if set(self.actual_columns) !=
                         set(self.expected_columns) else "column order differs")
        for r in self.marginals:
            if r["present"]:
                fails.extend(_marginal_failures(r, t))
        if self.correlations:
            for r in self.correlations:
                limit = max(t[f"corr_{r['kind']}"], t["corr_z"] * r["se"])
                if r["abs_diff"] > limit:
                    fails.append(f"correlation {r['a']} x {r['b']}: |diff| {r['abs_diff']:.3f}"
                                 f" > {limit:.3f}")
            rmse = self.correlation_summary()["rmse"]
            rms_se = float(np.sqrt(np.mean([r["se"] ** 2 for r in self.correlations])))
            limit = max(t["corr_rmse"], t["corr_rmse_se"] * rms_se)
            if rmse > limit:
                fails.append(f"correlation RMSE {rmse:.4f} > {limit:.4f}")
        return CheckResult(not fails, fails)


def _jsonable(r: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in r.items():
        if isinstance(v, (np.floating, float)):
            v = None if math.isnan(v) else float(v)
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        out[k] = v
    return out


def _scale(n: int) -> float:
    return math.sqrt(REFERENCE_N / n) if 0 < n < REFERENCE_N else 1.0


def _marginal_failures(r: Mapping[str, Any], t: Mapping[str, float]) -> list[str]:
    name, fails = r["column"], []
    n = r["n_nonmissing_dummy"]
    p = r["missing_rate_profile"]
    limit = t["missing_sigma"] * math.sqrt(p * (1 - p) / r["n_rows"]) + t["missing_abs"] \
        + 1 / r["n_rows"]
    if abs(r["missing_rate_diff"]) > limit:
        fails.append(f"{name}: missing rate differs by {r['missing_rate_diff']:+.4f}")
    if r.get("unseen_values"):
        fails.append(f"{name}: {r['unseen_values']} values not in the profile")
    if r.get("constant_ok") is False:
        fails.append(f"{name}: constant column has other values")
    if r.get("unique_ok") is False:
        fails.append(f"{name}: identifiers are not unique")
    s = _scale(n)
    checks = [("mean_diff_sd", "mean"), ("grid_ks", "grid_ks"), ("max_abs_dp", "max_abs_dp"),
              ("tvd", "tvd")]
    checks.append(("iqr_rel_diff", "iqr") if r.get("heavy_tailed") else ("sd_rel_diff", "sd"))
    for key, tkey in checks:
        v = r.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)) \
                and abs(v) > t[tkey] * s:
            fails.append(f"{name}: {key} {v:+.4f} exceeds {t[tkey] * s:.4f}")
    return fails


# ---------------------------------------------------------------------------
# marginal metrics
# ---------------------------------------------------------------------------


def grid_ks(x: np.ndarray, grid: Quantiles | None) -> float | None:
    """Largest distance between the dummy CDF and the profile grid on its inner points.

    At a tied value ``q`` the dummy CDF jumps from ``F(q-)`` to ``F(q)``; any ``p`` inside
    that jump counts as a match, so point masses (for example many zeros) are handled.
    """
    if grid is None or len(x) == 0 or len(grid.probs) < 3:
        return None
    xs = np.sort(x)
    q = np.array(grid.values[1:-1])
    p = np.array(grid.probs[1:-1])
    lo = np.searchsorted(xs, q, side="left") / len(xs)
    hi = np.searchsorted(xs, q, side="right") / len(xs)
    return float(np.max(np.maximum(0.0, np.maximum(lo - p, p - hi))))


def _frequency_metrics(values: np.ndarray, levels: list, probs: list[float]
                       ) -> dict[str, Any]:
    n = len(values)
    lookup = {repr(lv): i for i, lv in enumerate(levels)}
    counts = np.zeros(len(levels))
    unseen = 0
    for v in values:
        i = lookup.get(repr(v))
        if i is None:
            unseen += 1
        else:
            counts[i] += 1
    freq = counts / n if n else counts
    diff = freq - np.array(probs)
    return {"max_abs_dp": float(np.abs(diff).max()) if n else None,
            "tvd": float(0.5 * np.abs(diff).sum()) if n else None,
            "unseen_values": unseen}


def _heavy_tailed(col: NumericColumn) -> bool:
    sq = col.summary_quantiles
    if sq is None:
        return False
    q = dict(zip(sq.probs, sq.values, strict=True))
    iqr = q[0.75] - q[0.25]
    return iqr <= 0 or (q[0.99] - q[0.01]) / iqr > HEAVY_TAIL_RATIO


def _numeric_metrics(col: NumericColumn, x: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if col.mode == "discrete":
        out.update(_frequency_metrics(x.tolist(), list(col.support), col.probabilities))
        return out
    out["grid_ks"] = grid_ks(x, col.grid)
    if len(x) > 1 and col.sd:
        out["mean_diff_sd"] = float((x.mean() - col.mean) / col.sd)
        out["heavy_tailed"] = _heavy_tailed(col)
        if out["heavy_tailed"]:
            q = dict(zip(col.summary_quantiles.probs, col.summary_quantiles.values,
                         strict=True))
            iqr = q[0.75] - q[0.25]
            if iqr > 0:
                d_iqr = np.subtract(*np.quantile(x, [0.75, 0.25]))
                out["iqr_rel_diff"] = float(d_iqr / iqr - 1)
        else:
            out["sd_rel_diff"] = float(x.std(ddof=1) / col.sd - 1)
    return out


def _categorical_values(col: CategoricalColumn, s: pd.Series,
                        inverse: Mapping[str, Any] | None) -> np.ndarray:
    values = s[s.notna()]
    if inverse is not None:
        values = values.map(lambda v: inverse.get(str(v), v))
    kind = col.value_kind
    try:
        return normalize_values(values, kind if kind == value_kind(values) else "str")
    except (TypeError, ValueError):
        return values.astype(object).to_numpy()


def _column_metrics(col: ColumnProfile, s: pd.Series, inverse: Mapping[str, Any] | None
                    ) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(col, NumericColumn):
        x = to_float_array(s)
        out.update(_numeric_metrics(col, x[~np.isnan(x)]))
    elif isinstance(col, CategoricalColumn):
        values = _categorical_values(col, s, inverse)
        if col.value_kind == "str":
            values = np.array([str(v) for v in values], dtype=object)
        out.update(_frequency_metrics(values.tolist(), list(col.levels), col.probabilities))
    elif isinstance(col, DateColumn):
        days, _ = date_time_parts(to_timestamps(s)[0])
        out["grid_ks"] = grid_ks(days[~np.isnan(days)], col.date_part.grid)
    elif isinstance(col, DatetimeColumn):
        days, secs = date_time_parts(to_timestamps(s)[0])
        ok = ~np.isnan(days)
        out["grid_ks"] = grid_ks(days[ok], col.date_part.grid)
        out["grid_ks_time"] = grid_ks(secs[ok], col.time_part.grid)
    elif isinstance(col, TimeColumn):
        secs = time_seconds(s)
        out["grid_ks"] = grid_ks(secs[~np.isnan(secs)], col.time_part.grid)
    elif isinstance(col, ConstantColumn):
        values = s.dropna().astype(str)
        out["constant_ok"] = bool(values.nunique() <= 1)
    elif isinstance(col, IdColumn):
        out["unique_ok"] = bool(s.dropna().is_unique)
    return out


# ---------------------------------------------------------------------------
# dependence
# ---------------------------------------------------------------------------


def _reprofile_types(profile: Profile, dummy: pd.DataFrame) -> dict[str, Any]:
    types: dict[str, Any] = {}
    for col in profile.columns:
        if col.name not in dummy.columns:
            continue
        if isinstance(col, CategoricalColumn):
            types[col.name] = {"type": col.type.value, "levels": list(col.levels)}
        elif col.type in _CONTINUOUS:
            types[col.name] = col.type.value
        else:
            types[col.name] = "excluded"
    return types


def _dummy_for_reprofile(profile: Profile, dummy: pd.DataFrame,
                         inverse: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Dummy restricted to profiled columns, levels mapped back and unseen values blanked."""
    out = {}
    for col in profile.columns:
        if col.name not in dummy.columns:
            continue
        s = dummy[col.name]
        if isinstance(col, CategoricalColumn):
            values = pd.Series(_categorical_values(col, s, inverse.get(col.name)),
                               index=s.index[s.notna()])
            if col.value_kind == "str":
                values = values.map(str)
            known = {repr(lv) for lv in col.levels}
            values = values[[repr(v) in known for v in values]]
            s = values.reindex(s.index).astype(object)
            if col.value_kind in ("int", "float", "bool"):
                s = s.astype({"int": "Int64", "float": "float64", "bool": "boolean"}
                             [col.value_kind])
        out[col.name] = s
    return pd.DataFrame(out, index=dummy.index)


def _label_kind(profile: Profile, column: str) -> str:
    t = profile.column(column).type
    if t is ColumnType.NOMINAL:
        return "nominal"
    return "discrete" if t in CATEGORICAL_TYPES else "continuous"


def _pair_kind(a: str, b: str) -> str:
    kinds = {a, b}
    for k in ("nominal", "discrete"):
        if k in kinds:
            return k
    return "continuous"


def _tied_cells(grid: Quantiles | None, split: int = 50) -> np.ndarray:
    """Latent cell masses implied by a quantile grid: tied runs are single cells."""
    if grid is None:
        return np.array([1.0])
    probs, values = np.array(grid.probs), np.array(grid.values)
    cells: list[float] = []
    i = 0
    while i < len(values) - 1:
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        if j > i:
            cells.append(probs[j] - probs[i])
            i = j
        else:
            cells.extend([(probs[i + 1] - probs[i]) / split] * split)
            i += 1
    cells_arr = np.array([c for c in cells if c > 0])
    return cells_arr / cells_arr.sum()


def _efficiencies(profile: Profile) -> list[float]:
    """Variance of each latent dimension's normal scores (1 = no information lost)."""
    out: list[float] = []
    for col in profile.columns:
        if not col.has_latent:
            continue
        if isinstance(col, CategoricalColumn):
            if col.type is ColumnType.NOMINAL:
                cells_list = [np.array([1 - p, p]) for p in col.probabilities]
            else:
                cells_list = [np.array(col.probabilities)]
        elif isinstance(col, NumericColumn):
            cells_list = [np.array(col.probabilities) if col.mode == "discrete"
                          else _tied_cells(col.grid)]
        elif isinstance(col, DatetimeColumn):
            cells_list = [_tied_cells(col.date_part.grid), _tied_cells(col.time_part.grid)]
        elif isinstance(col, DateColumn):
            cells_list = [_tied_cells(col.date_part.grid)]
        else:
            cells_list = [_tied_cells(col.time_part.grid)]
        for cells in cells_list:
            cells = cells[cells > 0]
            scores, _ = interval_scores(cells)
            out.append(float(np.dot(cells, scores**2)))
    return out


def _correlations(profile: Profile, dummy_profile: Profile) -> list[dict[str, Any]]:
    labels = profile.latent_labels()
    eff = _efficiencies(profile)
    dep = profile.dependence
    d_labels = {lb.key: i for i, lb in enumerate(dummy_profile.latent_labels())}
    d_dep = dummy_profile.dependence
    min_pairs = profile.options["min_pair_count"]
    records = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            pv = dep.observed.matrix[i][j]
            if pv is None or dep.observed.n_pairs[i][j] < min_pairs:
                continue
            di, dj = d_labels.get(a.key), d_labels.get(b.key)
            if di is None or dj is None:
                continue
            dv = d_dep.observed.matrix[di][dj]
            n_pairs = d_dep.observed.n_pairs[di][dj]
            if dv is None or n_pairs < min_pairs:
                continue
            pl, dl = dep.latent.matrix[i][j], d_dep.latent.matrix[di][dj]
            records.append({
                "a": a.key, "b": b.key, "column_a": a.column, "column_b": b.column,
                "kind": _pair_kind(_label_kind(profile, a.column),
                                   _label_kind(profile, b.column)),
                "profile": pv, "dummy": dv, "abs_diff": abs(dv - pv),
                "profile_latent": pl, "dummy_latent": dl,
                "latent_abs_diff": None if pl is None or dl is None else abs(dl - pl),
                "n_pairs_dummy": int(n_pairs),
                # approximate sampling SE of the dummy's estimate (latent Gaussian model)
                "se": (1 - pl**2 if pl is not None else 1.0)
                / math.sqrt(max(n_pairs * eff[i] * eff[j], 1.0)),
            })
    return records


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _inverse_level_map(level_map: LevelMap | dict | str | Path | None
                       ) -> dict[str, dict[str, Any]]:
    if level_map is None:
        return {}
    if isinstance(level_map, dict):
        level_map = LevelMap.from_dict(level_map)
    elif not isinstance(level_map, LevelMap):
        level_map = LevelMap.load(level_map)
    return {col: {str(real): pseudo for pseudo, real in m.items()}
            for col, m in level_map.columns.items()}


def compare(profile: Profile, dummy: pd.DataFrame, *,
            level_map: LevelMap | dict | str | Path | None = None) -> CompareReport:
    """Compare ``dummy`` with ``profile`` column by column and pair by pair.

    Marginals are compared directly against the stored statistics (missing rate, mean and
    SD or IQR, grid CDF distance, level frequencies). Dependence is compared by profiling
    ``dummy`` again with the profile's column types and levels fixed and taking the
    difference of the latent correlation matrices (both the ``observed`` matrix used for
    checks and the generation-space ``latent`` matrix).

    Pass ``level_map`` if the dummy was generated with real level names so they can be
    mapped back to the pseudonyms stored in the profile.
    """
    from .profile import profile_dataframe  # local import: profile imports generate deps

    inverse = _inverse_level_map(level_map)
    n = len(dummy)
    marginals = []
    for col in profile.columns:
        if not col.in_output:
            continue
        present = col.name in dummy.columns
        record: dict[str, Any] = {"column": col.name, "type": col.type.value,
                                  "present": present, "n_rows": n,
                                  "missing_rate_profile": col.missing_rate}
        if present:
            s = dummy[col.name]
            n_obs = int(s.notna().sum())
            record.update(missing_rate_dummy=1 - n_obs / n if n else float("nan"),
                          n_nonmissing_dummy=n_obs)
            record["missing_rate_diff"] = record["missing_rate_dummy"] - col.missing_rate
            record.update(_column_metrics(col, s, inverse.get(col.name)))
        marginals.append(record)

    correlations: list[dict[str, Any]] = []
    if profile.latent_labels() and n:
        opts = profile.options
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dummy_profile = profile_dataframe(
                _dummy_for_reprofile(profile, dummy, inverse),
                types=_reprofile_types(profile, dummy), rare_threshold=1,
                min_tail_count=opts["min_tail_count"], quantile_grid=opts["quantile_grid"],
                discrete_max_levels=opts["discrete_max_levels"],
                max_levels=max(opts["max_levels"], _max_levels(profile)),
                ordinal_max_levels=opts["ordinal_max_levels"],
                min_pair_count=opts["min_pair_count"])
        correlations = _correlations(profile, dummy_profile)

    expected = profile.output_columns
    return CompareReport(n_rows=n, expected_columns=list(expected),
                         actual_columns=[str(c) for c in dummy.columns],
                         marginals=marginals, correlations=correlations)


def _max_levels(profile: Profile) -> int:
    return max([len(c.levels) for c in profile.columns if isinstance(c, CategoricalColumn)]
               or [0])
