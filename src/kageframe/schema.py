"""Profile JSON schema (version 0.1.0): dataclasses, validation, save/load.

A profile contains aggregate statistics only. The pseudonym-to-real-name level map
is held in memory (``Profile.level_map``) and is never written by ``Profile.save``;
it is saved separately with ``Profile.save_level_map`` for local use only.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ._version import __version__
from .exceptions import KageFrameWarning, ProfileSchemaError
from .types import CATEGORICAL_TYPES, LATENT_TYPES, ColumnType

SCHEMA_VERSION = "0.1.0"
LEVEL_MAP_KIND = "kageframe_level_map"
LEVEL_MAP_WARNING = "LOCAL ONLY - DO NOT SHARE. Maps pseudonymized level names to real values."
DATE_REFERENCE = "1970-01-01"

DEFAULT_OPTIONS: dict[str, Any] = {
    "rare_threshold": 10,
    "min_tail_count": 10,
    "quantile_grid": 101,
    "discrete_max_levels": 20,
    "max_levels": 50,
    "ordinal_max_levels": 10,
    "min_pair_count": 30,
    "datetime_mode": "split",
    "correlation_estimator": "interval_scores_mehler",
}

Scalar = str | int | float | bool


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _err(path: str, message: str) -> ProfileSchemaError:
    return ProfileSchemaError(f"{path}: {message}")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, bool)) or _is_number(v)


def _require_mapping(d: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(d, Mapping):
        raise _err(path, f"expected an object, got {type(d).__name__}")
    return d


def _check_keys(d: Mapping[str, Any], path: str, required: set[str], optional: set[str]) -> None:
    missing = required - set(d)
    if missing:
        raise _err(path, f"missing required fields {sorted(missing)}")
    unknown = set(d) - required - optional
    if unknown:
        raise _err(path, f"unknown fields {sorted(unknown)}")


def _number(v: Any, path: str, *, lo: float | None = None, hi: float | None = None,
            nullable: bool = False) -> float | None:
    if v is None and nullable:
        return None
    if not _is_number(v):
        raise _err(path, f"expected a finite number, got {v!r}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise _err(path, f"value {v!r} outside [{lo}, {hi}]")
    return v


def _integer(v: Any, path: str, *, lo: int | None = 0, nullable: bool = False) -> int | None:
    if v is None and nullable:
        return None
    if not isinstance(v, int) or isinstance(v, bool):
        raise _err(path, f"expected an integer, got {v!r}")
    if lo is not None and v < lo:
        raise _err(path, f"value {v!r} must be >= {lo}")
    return v


def _string(v: Any, path: str, *, nullable: bool = False,
            choices: Sequence[str] | None = None) -> str | None:
    if v is None and nullable:
        return None
    if not isinstance(v, str):
        raise _err(path, f"expected a string, got {v!r}")
    if choices is not None and v not in choices:
        raise _err(path, f"expected one of {list(choices)}, got {v!r}")
    return v


def _boolean(v: Any, path: str) -> bool:
    if not isinstance(v, bool):
        raise _err(path, f"expected a boolean, got {v!r}")
    return v


def _number_list(v: Any, path: str, *, nondecreasing: bool = False) -> list[float]:
    if not isinstance(v, list):
        raise _err(path, "expected a list")
    for i, x in enumerate(v):
        _number(x, f"{path}[{i}]")
    if nondecreasing and any(b < a for a, b in zip(v, v[1:], strict=False)):
        raise _err(path, "values must be non-decreasing")
    return list(v)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


@dataclass
class Quantiles:
    """Quantile function on a probability grid: ``values[i] = Q(probs[i])``."""

    probs: list[float]
    values: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {"probs": list(self.probs), "values": list(self.values)}

    @classmethod
    def from_dict(cls, d: Any, path: str) -> Quantiles:
        d = _require_mapping(d, path)
        _check_keys(d, path, {"probs", "values"}, set())
        probs = _number_list(d["probs"], f"{path}.probs")
        values = _number_list(d["values"], f"{path}.values", nondecreasing=True)
        if len(probs) != len(values) or len(probs) < 2:
            raise _err(path, "probs and values must have the same length (>= 2)")
        if any(p < 0 or p > 1 for p in probs) or any(
            b <= a for a, b in zip(probs, probs[1:], strict=False)
        ):
            raise _err(f"{path}.probs", "must be strictly increasing within [0, 1]")
        return cls(probs, values)


@dataclass(frozen=True)
class LatentLabel:
    """One dimension of the latent / observed-feature correlation matrices."""

    column: str
    component: str  # "value" | "level" | "date" | "time"
    level: Scalar | None = None

    @property
    def key(self) -> str:
        if self.component == "level":
            return f"{self.column}[{self.level}]"
        if self.component in ("date", "time"):
            return f"{self.column}::{self.component}"
        return self.column

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"column": self.column, "component": self.component}
        if self.component == "level":
            d["level"] = self.level
        d["key"] = self.key
        return d

    @classmethod
    def from_dict(cls, d: Any, path: str) -> LatentLabel:
        d = _require_mapping(d, path)
        _check_keys(d, path, {"column", "component", "key"}, {"level"})
        component = _string(d["component"], f"{path}.component",
                            choices=("value", "level", "date", "time"))
        level = d.get("level")
        if component == "level" and not _is_scalar(level):
            raise _err(f"{path}.level", "level labels need a scalar 'level'")
        label = cls(_string(d["column"], f"{path}.column"), component, level)
        if d["key"] != label.key:
            raise _err(f"{path}.key", f"expected {label.key!r}, got {d['key']!r}")
        return label


@dataclass
class WarningRecord:
    column: str | None
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "code": self.code, "message": self.message}

    @classmethod
    def from_dict(cls, d: Any, path: str) -> WarningRecord:
        d = _require_mapping(d, path)
        _check_keys(d, path, {"column", "code", "message"}, set())
        return cls(_string(d["column"], f"{path}.column", nullable=True),
                   _string(d["code"], f"{path}.code"), _string(d["message"], f"{path}.message"))


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------

_COMMON = {"name", "type", "source_dtype", "n_nonmissing", "missing_rate"}


@dataclass
class ColumnProfile:
    """Fields shared by every column. Subclasses add type-specific fields."""

    name: str
    type: ColumnType
    source_dtype: str
    n_nonmissing: int
    missing_rate: float

    TYPES: ClassVar[frozenset[ColumnType]] = frozenset()
    REQUIRED: ClassVar[set[str]] = set()
    OPTIONAL: ClassVar[set[str]] = set()

    def _common_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type.value, "source_dtype": self.source_dtype,
                "n_nonmissing": self.n_nonmissing, "missing_rate": self.missing_rate}

    def to_dict(self) -> dict[str, Any]:
        return self._common_dict()

    @property
    def has_latent(self) -> bool:
        return self.type in LATENT_TYPES

    @property
    def in_output(self) -> bool:
        return self.type not in (ColumnType.TEXT, ColumnType.EXCLUDED)

    @staticmethod
    def _common_from_dict(d: Mapping[str, Any], path: str) -> dict[str, Any]:
        return {
            "name": _string(d["name"], f"{path}.name"),
            "type": ColumnType(d["type"]),
            "source_dtype": _string(d["source_dtype"], f"{path}.source_dtype"),
            "n_nonmissing": _integer(d["n_nonmissing"], f"{path}.n_nonmissing"),
            "missing_rate": _number(d["missing_rate"], f"{path}.missing_rate", lo=0, hi=1),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], path: str) -> ColumnProfile:
        _check_keys(d, path, _COMMON | cls.REQUIRED, cls.OPTIONAL)
        return cls(**cls._common_from_dict(d, path), **cls._fields_from_dict(d, path))

    @classmethod
    def _fields_from_dict(cls, d: Mapping[str, Any], path: str) -> dict[str, Any]:
        return {}


@dataclass
class NumericColumn(ColumnProfile):
    """``min``/``max`` hold the k/n and 1-k/n quantiles, never the true extremes."""

    subtype: str = "float"
    decimals: int | None = 0  # None: more than 6 decimals, values are not rounded
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    sd: float | None = None
    summary_quantiles: Quantiles | None = None
    mode: str = "quantile"
    grid: Quantiles | None = None
    support: list[float] | None = None
    probabilities: list[float] | None = None

    TYPES = frozenset({ColumnType.NUMERIC})
    REQUIRED = {"subtype", "decimals", "min", "max", "mean", "sd", "summary_quantiles", "mode"}
    OPTIONAL = {"grid", "support", "probabilities"}

    def to_dict(self) -> dict[str, Any]:
        d = self._common_dict()
        d.update(subtype=self.subtype, decimals=self.decimals, min=self.min, max=self.max,
                 mean=self.mean, sd=self.sd,
                 summary_quantiles=self.summary_quantiles.to_dict()
                 if self.summary_quantiles else None,
                 mode=self.mode)
        if self.mode == "quantile":
            d["grid"] = self.grid.to_dict() if self.grid else None
        else:
            d["support"] = list(self.support or [])
            d["probabilities"] = list(self.probabilities or [])
        return d

    @classmethod
    def _fields_from_dict(cls, d: Mapping[str, Any], path: str) -> dict[str, Any]:
        mode = _string(d["mode"], f"{path}.mode", choices=("quantile", "discrete"))
        out: dict[str, Any] = {
            "subtype": _string(d["subtype"], f"{path}.subtype", choices=("integer", "float")),
            "decimals": _integer(d["decimals"], f"{path}.decimals", nullable=True),
            "min": _number(d["min"], f"{path}.min", nullable=True),
            "max": _number(d["max"], f"{path}.max", nullable=True),
            "mean": _number(d["mean"], f"{path}.mean", nullable=True),
            "sd": _number(d["sd"], f"{path}.sd", lo=0, nullable=True),
            "summary_quantiles": None if d["summary_quantiles"] is None
            else Quantiles.from_dict(d["summary_quantiles"], f"{path}.summary_quantiles"),
            "mode": mode,
        }
        if mode == "quantile":
            if "support" in d or "probabilities" in d:
                raise _err(path, "quantile mode must not have support/probabilities")
            grid = d.get("grid")
            out["grid"] = None if grid is None else Quantiles.from_dict(grid, f"{path}.grid")
        else:
            if "grid" in d:
                raise _err(path, "discrete mode must not have a grid")
            support = _number_list(d.get("support"), f"{path}.support")
            probs = _number_list(d.get("probabilities"), f"{path}.probabilities")
            if len(support) != len(probs) or len(set(support)) != len(support):
                raise _err(path, "support must be unique and match probabilities in length")
            _check_probabilities(probs, f"{path}.probabilities")
            out.update(support=support, probabilities=probs)
        if out["min"] is not None and out["max"] is not None and out["min"] > out["max"]:
            raise _err(path, "min must be <= max")
        return out


def _check_probabilities(probs: Sequence[float], path: str) -> None:
    if any(p < 0 or p > 1 for p in probs):
        raise _err(path, "probabilities must lie in [0, 1]")
    total = sum(probs)
    if probs and abs(total - 1) > 1e-6:
        raise _err(path, f"probabilities must sum to 1 (got {total:.8f})")


@dataclass
class CategoricalColumn(ColumnProfile):
    """binary / ordinal / nominal. ``levels`` are pseudonyms when ``pseudonymized``."""

    levels: list[Scalar] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    value_kind: str = "str"
    pseudonymized: bool = False
    thresholds: list[float] | None = None
    other_level: str | None = None
    n_merged_levels: int = 0

    TYPES = CATEGORICAL_TYPES
    REQUIRED = {"levels", "counts", "probabilities", "value_kind", "pseudonymized"}
    OPTIONAL = {"thresholds", "other_level", "n_merged_levels"}

    def to_dict(self) -> dict[str, Any]:
        d = self._common_dict()
        d.update(levels=list(self.levels), counts=list(self.counts),
                 probabilities=list(self.probabilities), value_kind=self.value_kind,
                 pseudonymized=self.pseudonymized)
        if self.type is ColumnType.NOMINAL:
            d.update(other_level=self.other_level, n_merged_levels=self.n_merged_levels)
        else:
            d["thresholds"] = list(self.thresholds or [])
        return d

    @classmethod
    def _fields_from_dict(cls, d: Mapping[str, Any], path: str) -> dict[str, Any]:
        t = ColumnType(d["type"])
        levels = d["levels"]
        if not isinstance(levels, list) or not all(_is_scalar(v) for v in levels):
            raise _err(f"{path}.levels", "expected a list of scalars")
        if len(set(map(repr, levels))) != len(levels):
            raise _err(f"{path}.levels", "levels must be unique")
        k = len(levels)
        if t is ColumnType.BINARY and k != 2:
            raise _err(f"{path}.levels", "binary columns need exactly 2 levels")
        if k < 2:
            raise _err(f"{path}.levels", "need at least 2 levels")
        counts = d["counts"]
        if not isinstance(counts, list) or len(counts) != k:
            raise _err(f"{path}.counts", "expected one count per level")
        for i, c in enumerate(counts):
            _integer(c, f"{path}.counts[{i}]")
        probs = _number_list(d["probabilities"], f"{path}.probabilities")
        if len(probs) != k:
            raise _err(f"{path}.probabilities", "expected one probability per level")
        _check_probabilities(probs, f"{path}.probabilities")
        out: dict[str, Any] = {
            "levels": list(levels), "counts": list(counts), "probabilities": probs,
            "value_kind": _string(d["value_kind"], f"{path}.value_kind",
                                  choices=("str", "int", "float", "bool")),
            "pseudonymized": _boolean(d["pseudonymized"], f"{path}.pseudonymized"),
        }
        if t is ColumnType.NOMINAL:
            if "thresholds" in d:
                raise _err(path, "nominal columns have no thresholds")
            other = _string(d.get("other_level"), f"{path}.other_level", nullable=True)
            if other is not None and other not in levels:
                raise _err(f"{path}.other_level", "must be one of the levels")
            out.update(other_level=other,
                       n_merged_levels=_integer(d.get("n_merged_levels", 0),
                                                f"{path}.n_merged_levels"))
        else:
            if "other_level" in d or "n_merged_levels" in d:
                raise _err(path, "other_level/n_merged_levels are nominal-only")
            thresholds = _number_list(d.get("thresholds"), f"{path}.thresholds",
                                      nondecreasing=True)
            if len(thresholds) != k - 1:
                raise _err(f"{path}.thresholds", f"expected {k - 1} thresholds")
            out["thresholds"] = thresholds
        return out


@dataclass
class TemporalPart:
    """Numeric representation of a date (days since 1970-01-01) or time (seconds)."""

    min: float | None
    max: float | None
    grid: Quantiles | None
    resolution_seconds: int | None = None

    def to_dict(self, with_resolution: bool) -> dict[str, Any]:
        d: dict[str, Any] = {"min": self.min, "max": self.max,
                             "grid": self.grid.to_dict() if self.grid else None}
        if with_resolution:
            d["resolution_seconds"] = self.resolution_seconds
        return d

    @classmethod
    def from_dict(cls, d: Any, path: str, with_resolution: bool) -> TemporalPart:
        d = _require_mapping(d, path)
        _check_keys(d, path, {"min", "max", "grid"} | ({"resolution_seconds"}
                                                       if with_resolution else set()), set())
        grid = None if d["grid"] is None else Quantiles.from_dict(d["grid"], f"{path}.grid")
        lo = _number(d["min"], f"{path}.min", nullable=True)
        hi = _number(d["max"], f"{path}.max", nullable=True)
        if lo is not None and hi is not None and lo > hi:
            raise _err(path, "min must be <= max")
        res = None
        if with_resolution:
            res = _integer(d["resolution_seconds"], f"{path}.resolution_seconds", lo=1)
        return cls(lo, hi, grid, res)


@dataclass
class DateColumn(ColumnProfile):
    date_part: TemporalPart = field(default_factory=lambda: TemporalPart(None, None, None))
    output_format: str | None = None

    TYPES = frozenset({ColumnType.DATE})
    REQUIRED = {"reference", "unit", "date_part", "output_format"}

    def to_dict(self) -> dict[str, Any]:
        d = self._common_dict()
        d.update(reference=DATE_REFERENCE, unit="days",
                 date_part=self.date_part.to_dict(False), output_format=self.output_format)
        return d

    @classmethod
    def _fields_from_dict(cls, d: Mapping[str, Any], path: str) -> dict[str, Any]:
        _string(d["reference"], f"{path}.reference", choices=(DATE_REFERENCE,))
        _string(d["unit"], f"{path}.unit", choices=("days",))
        return {"date_part": TemporalPart.from_dict(d["date_part"], f"{path}.date_part", False),
                "output_format": _string(d["output_format"], f"{path}.output_format",
                                         nullable=True)}


@dataclass
class TimeColumn(ColumnProfile):
    time_part: TemporalPart = field(default_factory=lambda: TemporalPart(None, None, None, 1))
    output_format: str | None = None

    TYPES = frozenset({ColumnType.TIME})
    REQUIRED = {"unit", "time_part", "output_format"}

    def to_dict(self) -> dict[str, Any]:
        d = self._common_dict()
        d.update(unit="seconds_from_midnight", time_part=self.time_part.to_dict(True),
                 output_format=self.output_format)
        return d

    @classmethod
    def _fields_from_dict(cls, d: Mapping[str, Any], path: str) -> dict[str, Any]:
        _string(d["unit"], f"{path}.unit", choices=("seconds_from_midnight",))
        return {"time_part": TemporalPart.from_dict(d["time_part"], f"{path}.time_part", True),
                "output_format": _string(d["output_format"], f"{path}.output_format",
                                         nullable=True)}


@dataclass
class DatetimeColumn(ColumnProfile):
    """Schema 0.1.0 supports only the split representation (date part + time part)."""

    date_part: TemporalPart = field(default_factory=lambda: TemporalPart(None, None, None))
    time_part: TemporalPart = field(default_factory=lambda: TemporalPart(None, None, None, 1))
    tz: str | None = None
    output_format: str | None = None

    TYPES = frozenset({ColumnType.DATETIME})
    REQUIRED = {"reference", "date_part", "time_part", "tz", "output_format"}

    def to_dict(self) -> dict[str, Any]:
        d = self._common_dict()
        d.update(reference=DATE_REFERENCE, date_part=self.date_part.to_dict(False),
                 time_part=self.time_part.to_dict(True), tz=self.tz,
                 output_format=self.output_format)
        return d

    @classmethod
    def _fields_from_dict(cls, d: Mapping[str, Any], path: str) -> dict[str, Any]:
        _string(d["reference"], f"{path}.reference", choices=(DATE_REFERENCE,))
        return {
            "date_part": TemporalPart.from_dict(d["date_part"], f"{path}.date_part", False),
            "time_part": TemporalPart.from_dict(d["time_part"], f"{path}.time_part", True),
            "tz": _string(d["tz"], f"{path}.tz", nullable=True),
            "output_format": _string(d["output_format"], f"{path}.output_format", nullable=True),
        }


@dataclass
class IdColumn(ColumnProfile):
    """Identifier column: no statistics are kept; new sequential ids are generated."""

    value_kind: str = "str"
    reason: str = ""
    prefix: str = "ID"
    width: int = 6

    TYPES = frozenset({ColumnType.ID})
    REQUIRED = {"reason", "value_kind", "regenerate"}

    def to_dict(self) -> dict[str, Any]:
        d = self._common_dict()
        d.update(reason=self.reason, value_kind=self.value_kind,
                 regenerate={"kind": "sequential", "prefix": self.prefix, "width": self.width})
        return d

    @classmethod
    def _fields_from_dict(cls, d: Mapping[str, Any], path: str) -> dict[str, Any]:
        regen = _require_mapping(d["regenerate"], f"{path}.regenerate")
        _check_keys(regen, f"{path}.regenerate", {"kind", "prefix", "width"}, set())
        _string(regen["kind"], f"{path}.regenerate.kind", choices=("sequential",))
        return {"reason": _string(d["reason"], f"{path}.reason"),
                "value_kind": _string(d["value_kind"], f"{path}.value_kind",
                                      choices=("int", "str")),
                "prefix": _string(regen["prefix"], f"{path}.regenerate.prefix"),
                "width": _integer(regen["width"], f"{path}.regenerate.width", lo=1)}


@dataclass
class ConstantColumn(ColumnProfile):
    value: Scalar = ""
    value_kind: str = "str"

    TYPES = frozenset({ColumnType.CONSTANT})
    REQUIRED = {"value", "value_kind"}

    def to_dict(self) -> dict[str, Any]:
        d = self._common_dict()
        d.update(value=self.value, value_kind=self.value_kind)
        return d

    @classmethod
    def _fields_from_dict(cls, d: Mapping[str, Any], path: str) -> dict[str, Any]:
        if not _is_scalar(d["value"]):
            raise _err(f"{path}.value", "expected a scalar")
        return {"value": d["value"],
                "value_kind": _string(d["value_kind"], f"{path}.value_kind",
                                      choices=("str", "int", "float", "bool", "date",
                                               "datetime", "time"))}


@dataclass
class ReasonColumn(ColumnProfile):
    """text / empty / excluded: only the reason is kept."""

    reason: str = ""

    TYPES = frozenset({ColumnType.TEXT, ColumnType.EMPTY, ColumnType.EXCLUDED})
    REQUIRED = {"reason"}

    def to_dict(self) -> dict[str, Any]:
        d = self._common_dict()
        d["reason"] = self.reason
        return d

    @classmethod
    def _fields_from_dict(cls, d: Mapping[str, Any], path: str) -> dict[str, Any]:
        return {"reason": _string(d["reason"], f"{path}.reason")}


_COLUMN_CLASSES: dict[ColumnType, type[ColumnProfile]] = {
    t: cls
    for cls in (NumericColumn, CategoricalColumn, DateColumn, TimeColumn, DatetimeColumn,
                IdColumn, ConstantColumn, ReasonColumn)
    for t in cls.TYPES
}


def column_from_dict(d: Any, path: str) -> ColumnProfile:
    d = _require_mapping(d, path)
    try:
        t = ColumnType(d.get("type"))
    except ValueError:
        raise _err(f"{path}.type", f"unknown column type {d.get('type')!r}") from None
    return _COLUMN_CLASSES[t].from_dict(d, path)


# ---------------------------------------------------------------------------
# latent labels and dependence
# ---------------------------------------------------------------------------


def latent_labels(columns: Sequence[ColumnProfile]) -> list[LatentLabel]:
    """Latent dimensions in profile column order.

    numeric/binary/ordinal/date/time: one ``value`` dimension; nominal: one ``level``
    dimension per level in ``levels`` order; datetime: ``date`` then ``time``.
    Columns without a latent representation are skipped.
    """
    labels: list[LatentLabel] = []
    for col in columns:
        if not col.has_latent:
            continue
        if col.type is ColumnType.NOMINAL:
            assert isinstance(col, CategoricalColumn)
            labels.extend(LatentLabel(col.name, "level", lv) for lv in col.levels)
        elif col.type is ColumnType.DATETIME:
            labels.extend([LatentLabel(col.name, "date"), LatentLabel(col.name, "time")])
        else:
            labels.append(LatentLabel(col.name, "value"))
    return labels


def _square_matrix(m: Any, d: int, path: str, *, allow_null: bool,
                   integer: bool = False) -> list[list[Any]]:
    if not isinstance(m, list) or len(m) != d:
        raise _err(path, f"expected a {d}x{d} matrix")
    for i, row in enumerate(m):
        if not isinstance(row, list) or len(row) != d:
            raise _err(f"{path}[{i}]", f"expected a row of length {d}")
        for j, v in enumerate(row):
            if v is None and allow_null:
                continue
            if integer:
                _integer(v, f"{path}[{i}][{j}]")
            else:
                _number(v, f"{path}[{i}][{j}]", lo=-1, hi=1)
    for i in range(d):
        for j in range(i + 1, d):
            if m[i][j] != m[j][i]:
                raise _err(path, f"matrix must be symmetric (entry [{i}][{j}])")
    return m


@dataclass
class CorrelationMatrix:
    labels: list[LatentLabel]
    matrix: list[list[float | None]]
    n_pairs: list[list[int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"labels": [lb.to_dict() for lb in self.labels],
                             "matrix": [list(r) for r in self.matrix]}
        if self.n_pairs is not None:
            d["n_pairs"] = [list(r) for r in self.n_pairs]
        return d


@dataclass
class Dependence:
    """``observed``: one-hot feature space (compare target); ``latent``: generation space."""

    observed: CorrelationMatrix
    latent: CorrelationMatrix
    method: str = "latent_gaussian_copula"
    psd_correction: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "observed": self.observed.to_dict(),
                "latent": self.latent.to_dict(), "psd_correction": dict(self.psd_correction)}

    @classmethod
    def from_dict(cls, d: Any, path: str, expected: list[LatentLabel]) -> Dependence:
        d = _require_mapping(d, path)
        _check_keys(d, path, {"method", "observed", "latent", "psd_correction"}, set())
        _string(d["method"], f"{path}.method", choices=("latent_gaussian_copula",))
        dim = len(expected)
        blocks = nominal_blocks(expected)
        parsed = {}
        for part in ("observed", "latent"):
            p = f"{path}.{part}"
            m = _require_mapping(d[part], p)
            _check_keys(m, p, {"labels", "matrix"}, {"n_pairs"} if part == "observed" else set())
            if not isinstance(m["labels"], list):
                raise _err(f"{p}.labels", "expected a list")
            labels = [LatentLabel.from_dict(x, f"{p}.labels[{i}]")
                      for i, x in enumerate(m["labels"])]
            if labels != expected:
                raise _err(f"{p}.labels", "labels do not match the latent dimensions implied "
                                          "by columns")
            matrix = _square_matrix(m["matrix"], dim, f"{p}.matrix",
                                    allow_null=part == "observed")
            n_pairs = None
            if part == "observed":
                if "n_pairs" not in m:
                    raise _err(p, "missing required fields ['n_pairs']")
                n_pairs = _square_matrix(m["n_pairs"], dim, f"{p}.n_pairs", allow_null=False,
                                         integer=True)
            _check_blocks(matrix, blocks, dim, p, latent=part == "latent")
            parsed[part] = CorrelationMatrix(labels, matrix, n_pairs)
        psd = _require_mapping(d["psd_correction"], f"{path}.psd_correction")
        return cls(parsed["observed"], parsed["latent"], d["method"], dict(psd))


def nominal_blocks(labels: list[LatentLabel]) -> list[int]:
    """Block id per dimension; dimensions of the same nominal column share an id."""
    ids: list[int] = []
    for i, lb in enumerate(labels):
        if lb.component == "level" and i > 0 and labels[i - 1].component == "level" \
                and labels[i - 1].column == lb.column:
            ids.append(ids[-1])
        else:
            ids.append(i)
    return ids


def _check_blocks(m: list[list[Any]], blocks: list[int], d: int, path: str, *,
                  latent: bool) -> None:
    for i in range(d):
        if m[i][i] is None or abs(m[i][i] - 1) > 1e-9:
            raise _err(f"{path}.matrix[{i}][{i}]", "diagonal entries must be 1")
        for j in range(d):
            if i == j:
                continue
            same = blocks[i] == blocks[j]
            v = m[i][j]
            if latent and same and abs(v) > 1e-9:
                raise _err(f"{path}.matrix[{i}][{j}]",
                           "latent scores of the same nominal column must be uncorrelated")
            if not latent and v is None and not same:
                raise _err(f"{path}.matrix[{i}][{j}]",
                           "null is only allowed within a nominal column's block")


# ---------------------------------------------------------------------------
# level map (local only)
# ---------------------------------------------------------------------------


@dataclass
class LevelMap:
    """Pseudonym -> real level value, per column. Never part of the profile JSON."""

    columns: dict[str, dict[str, Scalar]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": LEVEL_MAP_KIND, "schema_version": SCHEMA_VERSION,
                "warning": LEVEL_MAP_WARNING,
                "columns": {c: dict(m) for c, m in self.columns.items()}}

    @classmethod
    def from_dict(cls, d: Any) -> LevelMap:
        d = _require_mapping(d, "level_map")
        if d.get("kind") != LEVEL_MAP_KIND:
            raise ProfileSchemaError("not a kageframe level map (missing kind="
                                     f"{LEVEL_MAP_KIND!r})")
        _check_keys(d, "level_map", {"kind", "schema_version", "warning", "columns"}, set())
        _check_version(d["schema_version"], "level_map.schema_version")
        cols = _require_mapping(d["columns"], "level_map.columns")
        out: dict[str, dict[str, Scalar]] = {}
        for c, m in cols.items():
            m = _require_mapping(m, f"level_map.columns.{c}")
            for k, v in m.items():
                if not _is_scalar(v):
                    raise _err(f"level_map.columns.{c}.{k}", "expected a scalar")
            if len(set(map(repr, m.values()))) != len(m):
                raise _err(f"level_map.columns.{c}", "real values must be unique")
            out[c] = dict(m)
        return cls(out)

    def save(self, path: str | Path) -> None:
        """Write the map as JSON. Keep the file local (``*.local.json`` is gitignored)."""
        Path(path).write_text(_dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LevelMap:
        """Read a map written by :meth:`save` or :meth:`Profile.save_level_map`."""
        return cls.from_dict(_loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


def _check_version(v: Any, path: str) -> None:
    v = _string(v, path)
    try:
        major, minor, *_ = (int(x) for x in v.split("."))
    except ValueError:
        raise _err(path, f"invalid version {v!r}") from None
    cur_major, cur_minor, *_ = (int(x) for x in SCHEMA_VERSION.split("."))
    if major != cur_major:
        raise _err(path, f"schema version {v} is incompatible with {SCHEMA_VERSION}")
    if minor > cur_minor:
        warnings.warn(f"profile schema {v} is newer than supported {SCHEMA_VERSION}",
                      KageFrameWarning, stacklevel=4)


def _check_options(d: Any) -> dict[str, Any]:
    d = _require_mapping(d, "options")
    unknown = set(d) - set(DEFAULT_OPTIONS)
    if unknown:
        raise _err("options", f"unknown options {sorted(unknown)}")
    out = dict(DEFAULT_OPTIONS)
    for k, v in d.items():
        default = DEFAULT_OPTIONS[k]
        if isinstance(default, bool):
            _boolean(v, f"options.{k}")
        elif isinstance(default, int):
            _integer(v, f"options.{k}", lo=1)
        else:
            _string(v, f"options.{k}")
        out[k] = v
    return out


def _reject_constant(token: str) -> Any:
    raise ProfileSchemaError(f"non-finite number {token} is not allowed in kageframe JSON")


def _dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    except ValueError as e:
        raise ProfileSchemaError(f"cannot serialize: {e}") from None


def _loads(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as e:
        raise ProfileSchemaError(f"invalid JSON: {e}") from None


@dataclass
class Profile:
    """Aggregate statistics of a DataFrame: everything :func:`generate` needs, no rows.

    Contains, per column, the type, missing rate and marginal distribution (blurred
    quantile grids, level frequencies), plus the latent correlation matrices and the
    warnings raised while profiling. Real row values, true minima/maxima, merged rare
    level names, identifiers and free text are never stored.

    Create it with :func:`profile_dataframe`; share it with :meth:`save` /
    :func:`load_profile`. ``level_map`` (pseudonym -> real name) lives only in memory
    and is written separately by :meth:`save_level_map`.
    """

    n_rows: int
    columns: list[ColumnProfile]
    options: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_OPTIONS))
    dependence: Dependence | None = None
    warnings: list[WarningRecord] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    kageframe_version: str = __version__
    level_map: LevelMap | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.validate()

    # -- structure ---------------------------------------------------------

    def column(self, name: str) -> ColumnProfile:
        """The profile of column ``name`` (``KeyError`` if unknown)."""
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    @property
    def column_names(self) -> list[str]:
        """All profiled columns in the original order."""
        return [c.name for c in self.columns]

    @property
    def output_columns(self) -> list[str]:
        """Columns that :func:`generate` returns (free text and excluded columns dropped)."""
        return [c.name for c in self.columns if c.in_output]

    def latent_labels(self) -> list[LatentLabel]:
        """Labels of the rows/columns of the correlation matrices."""
        return latent_labels(self.columns)

    def validate(self) -> None:
        """Re-validate by round-tripping through the dict representation."""
        _integer(self.n_rows, "n_rows", lo=1)
        names = self.column_names
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise _err("columns", f"duplicate column names {dupes}")
        for i, col in enumerate(self.columns):
            if col.type not in type(col).TYPES:
                raise _err(f"columns[{i}]", f"{type(col).__name__} cannot have type "
                                            f"{col.type.value!r}")
            if col.n_nonmissing > self.n_rows:
                raise _err(f"columns[{i}].n_nonmissing", "exceeds n_rows")
            column_from_dict(col.to_dict(), f"columns[{i}]")
        if self.dependence is not None:
            Dependence.from_dict(self.dependence.to_dict(), "dependence", self.latent_labels())
        if self.level_map is not None:
            for c, mapping in self.level_map.columns.items():
                col = self.column(c) if c in names else None
                if isinstance(col, CategoricalColumn) and col.pseudonymized:
                    known = set(map(str, col.levels))
                elif isinstance(col, ConstantColumn):
                    known = {str(col.value)}
                else:
                    raise _err("level_map", f"column {c!r} is not a pseudonymized categorical")
                unknown = set(mapping) - known
                if unknown:
                    raise _err(f"level_map.columns.{c}",
                               f"pseudonyms not among the column's levels: {sorted(unknown)}")

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The JSON-compatible representation (without the level map)."""
        return {
            "schema_version": self.schema_version,
            "kageframe_version": self.kageframe_version,
            "n_rows": self.n_rows,
            "options": dict(self.options),
            "columns": [c.to_dict() for c in self.columns],
            "dependence": self.dependence.to_dict() if self.dependence else None,
            "warnings": [w.to_dict() for w in self.warnings],
        }

    @classmethod
    def from_dict(cls, d: Any) -> Profile:
        """Build and validate a profile; raises :class:`ProfileSchemaError` if invalid."""
        d = _require_mapping(d, "profile")
        if d.get("kind") == LEVEL_MAP_KIND:
            raise ProfileSchemaError("this file is a level map, not a profile; "
                                     "load it with LevelMap.load")
        _check_keys(d, "profile", {"schema_version", "kageframe_version", "n_rows", "options",
                                   "columns", "dependence", "warnings"}, set())
        _check_version(d["schema_version"], "schema_version")
        if not isinstance(d["columns"], list):
            raise _err("columns", "expected a list")
        columns = [column_from_dict(c, f"columns[{i}]") for i, c in enumerate(d["columns"])]
        dependence = None
        if d["dependence"] is not None:
            dependence = Dependence.from_dict(d["dependence"], "dependence",
                                              latent_labels(columns))
        if not isinstance(d["warnings"], list):
            raise _err("warnings", "expected a list")
        return cls(
            n_rows=_integer(d["n_rows"], "n_rows", lo=1),
            columns=columns,
            options=_check_options(d["options"]),
            dependence=dependence,
            warnings=[WarningRecord.from_dict(w, f"warnings[{i}]")
                      for i, w in enumerate(d["warnings"])],
            schema_version=d["schema_version"],
            kageframe_version=_string(d["kageframe_version"], "kageframe_version"),
        )

    def to_json(self) -> str:
        """Deterministic JSON text: the same DataFrame always gives the same string."""
        return _dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> Profile:
        """Parse and validate JSON text written by :meth:`to_json`."""
        return cls.from_dict(_loads(text))

    def save(self, path: str | Path) -> None:
        """Write the shareable profile JSON. The level map is never included."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Profile:
        """Read a profile written by :meth:`save`."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save_level_map(self, path: str | Path) -> None:
        """Write the pseudonym -> real name map for local use only."""
        if self.level_map is None:
            raise ValueError("this profile has no level map (it was loaded from JSON or "
                             "no levels were pseudonymized)")
        self.level_map.save(path)


def load_profile(path: str | Path) -> Profile:
    """Read and validate a profile JSON file (same as :meth:`Profile.load`)."""
    return Profile.load(path)
