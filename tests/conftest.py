from __future__ import annotations

import warnings

import pandas as pd
import pytest

from clinmock import profile_dataframe
from clinmock.datasets import RECOMMENDED_TYPES, make_clinical_like_df
from clinmock.schema import (
    CategoricalColumn,
    ConstantColumn,
    CorrelationMatrix,
    DateColumn,
    DatetimeColumn,
    Dependence,
    IdColumn,
    LevelMap,
    NumericColumn,
    Profile,
    Quantiles,
    ReasonColumn,
    TemporalPart,
    TimeColumn,
    WarningRecord,
    latent_labels,
)
from clinmock.types import ColumnType as T


@pytest.fixture(scope="session")
def clinical_df() -> pd.DataFrame:
    return make_clinical_like_df(n=10000, seed=0, edge_cases=True)


@pytest.fixture(scope="session")
def clinical_profile(clinical_df: pd.DataFrame) -> Profile:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return profile_dataframe(clinical_df, types=RECOMMENDED_TYPES)


def _grid(values: list[float]) -> Quantiles:
    n = len(values)
    return Quantiles([i / (n - 1) for i in range(n)], values)


def build_example_columns() -> list:
    return [
        IdColumn("patient_id", T.ID, "str", 1000, 0.0, value_kind="str",
                 reason="mostly unique short strings"),
        NumericColumn("age", T.NUMERIC, "Int64", 990, 0.01, subtype="integer", decimals=0,
                      min=31.0, max=93.0, mean=65.2, sd=11.9,
                      summary_quantiles=_grid([31.0, 57.0, 65.0, 73.0, 93.0]),
                      mode="quantile", grid=_grid([31.0, 45.0, 57.0, 65.0, 73.0, 85.0, 93.0])),
        NumericColumn("pain_score", T.NUMERIC, "int64", 1000, 0.0, subtype="integer",
                      decimals=0, min=0.0, max=3.0, mean=1.1, sd=0.9,
                      summary_quantiles=_grid([0.0, 1.0, 3.0]), mode="discrete",
                      support=[0.0, 1.0, 2.0, 3.0], probabilities=[0.3, 0.4, 0.2, 0.1]),
        CategoricalColumn("sex", T.BINARY, "str", 1000, 0.0, levels=["L01", "L02"],
                          counts=[520, 480], probabilities=[0.52, 0.48], value_kind="str",
                          pseudonymized=True, thresholds=[0.0502]),
        CategoricalColumn("ecog", T.ORDINAL, "Int64", 920, 0.08, levels=[0, 1, 2, 3],
                          counts=[368, 276, 184, 92], probabilities=[0.4, 0.3, 0.2, 0.1],
                          value_kind="int", pseudonymized=False,
                          thresholds=[-0.2533, 0.5244, 1.2816]),
        CategoricalColumn("indication", T.NOMINAL, "str", 980, 0.02,
                          levels=["L01", "L02", "Other"], counts=[500, 460, 20],
                          probabilities=[500 / 980, 460 / 980, 20 / 980], value_kind="str",
                          pseudonymized=True, other_level="Other", n_merged_levels=2),
        DateColumn("exam_date", T.DATE, "datetime64[s]", 950, 0.05,
                   date_part=TemporalPart(17900.0, 19300.0,
                                          _grid([17900.0, 18500.0, 19300.0])),
                   output_format=None),
        DatetimeColumn("admit_dt", T.DATETIME, "datetime64[s]", 900, 0.1,
                       date_part=TemporalPart(17900.0, 19300.0,
                                              _grid([17900.0, 18600.0, 19300.0])),
                       time_part=TemporalPart(21600.0, 79200.0,
                                              _grid([21600.0, 46800.0, 79200.0]), 60),
                       tz="Asia/Tokyo", output_format=None),
        TimeColumn("visit_time", T.TIME, "str", 850, 0.15,
                   time_part=TemporalPart(28800.0, 64800.0,
                                          _grid([28800.0, 37800.0, 64800.0]), 60),
                   output_format="%H:%M"),
        ReasonColumn("note", T.TEXT, "str", 700, 0.3, reason="free text"),
        ConstantColumn("site", T.CONSTANT, "str", 1000, 0.0, value="A", value_kind="str"),
        ReasonColumn("unused", T.EMPTY, "float64", 0, 1.0, reason="all values missing"),
        ReasonColumn("dropped", T.EXCLUDED, "float64", 1000, 0.0, reason="user excluded"),
    ]


def build_example_dependence(columns: list) -> Dependence:
    labels = latent_labels(columns)
    d = len(labels)
    keys = [lb.key for lb in labels]
    latent = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
    observed = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
    ind = [keys.index(k) for k in ("indication[L01]", "indication[L02]", "indication[Other]")]
    for i in ind:
        for j in ind:
            if i != j:
                observed[i][j] = None
    pairs = [("age", "ecog", 0.35), ("age", "indication[L01]", 0.2),
             ("admit_dt::date", "admit_dt::time", 0.1), ("sex", "pain_score", -0.15)]
    for a, b, r in pairs:
        i, j = keys.index(a), keys.index(b)
        latent[i][j] = latent[j][i] = r
        observed[i][j] = observed[j][i] = r
    n_pairs = [[900] * d for _ in range(d)]
    return Dependence(
        observed=CorrelationMatrix(labels, observed, n_pairs),
        latent=CorrelationMatrix(labels, latent),
        psd_correction={"applied": False, "min_eigenvalue_before": 0.61,
                        "frobenius_change": 0.0, "iterations": 0},
    )


def build_example_profile(with_level_map: bool = True) -> Profile:
    columns = build_example_columns()
    return Profile(
        n_rows=1000,
        columns=columns,
        dependence=build_example_dependence(columns),
        warnings=[WarningRecord("patient_id", "id_like_excluded", "excluded")],
        level_map=LevelMap({"sex": {"L01": "M", "L02": "F"},
                            "indication": {"L01": "surveillance", "L02": "screening"}})
        if with_level_map else None,
    )


@pytest.fixture
def example_profile() -> Profile:
    return build_example_profile()
