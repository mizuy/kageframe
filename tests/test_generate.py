"""generate(): reconstruction of each column type from a profile.

The MVP acceptance criteria (columns, marginals, frequencies, missing rates, dependence,
nominal, reproducibility) are tested end to end in ``test_mvp_acceptance.py``; this file
covers the mechanics of generation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from kageframe import generate, profile_dataframe
from kageframe.datasets import RECOMMENDED_TYPES
from kageframe.exceptions import KageFrameWarning
from kageframe.latent import nominal_choice
from kageframe.schema import CategoricalColumn, Profile
from kageframe.types import ColumnType as T


@pytest.fixture(scope="module")
def generated(clinical_profile) -> pd.DataFrame:
    return generate(clinical_profile, seed=123)


def _quiet_profile(df: pd.DataFrame, **kwargs) -> Profile:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return profile_dataframe(df, **kwargs)


def test_default_n_and_explicit_n(clinical_df, clinical_profile, generated) -> None:
    assert len(generated) == len(clinical_df)
    assert len(generate(clinical_profile, n=250, seed=0)) == 250


def test_dtypes_follow_the_source(clinical_df, generated) -> None:
    for name in ("age", "smoker", "ecog"):
        assert generated[name].dtype == "Int64"  # integers with missing values
    assert generated["bmi"].dtype == np.float64
    assert generated["lab_flag"].dtype == "boolean"
    assert generated["exam_date"].dtype == clinical_df["exam_date"].dtype
    assert generated["admit_dt"].dtype == clinical_df["admit_dt"].dtype
    assert set(generated["sex"].dropna()) <= {"F", "M"}


def test_integer_column_without_missing_stays_int64() -> None:
    rng = np.random.default_rng(0)
    p = _quiet_profile(pd.DataFrame({"x": rng.integers(0, 1000, 2000)}),
                       types={"x": "numeric"})
    assert generate(p, seed=0)["x"].dtype == np.int64


def test_numeric_values_stay_inside_blurred_range(clinical_profile, generated) -> None:
    for name in ("age", "bmi", "crp"):
        col = clinical_profile.column(name)
        g = generated[name].dropna().to_numpy(dtype=float)
        assert g.min() >= col.min and g.max() <= col.max
        assert np.all(np.round(g, col.decimals) == g)


def test_discrete_numeric_support() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"pain": rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1], size=5000)})
    p = profile_dataframe(df, types={"pain": "numeric"})
    freq = generate(p, seed=0)["pain"].value_counts(normalize=True).sort_index()
    assert list(freq.index) == [0, 1, 2, 3]
    assert freq.to_numpy() == pytest.approx(p.column("pain").probabilities, abs=0.015)


def test_nominal_choice_picks_exactly_one_level() -> None:
    """Gumbel-max picks one index per row, reproducing the level probabilities."""
    p = np.array([0.5, 0.3, 0.15, 0.05])
    z = np.random.default_rng(0).standard_normal((100_000, 4))
    idx = nominal_choice(z, p)
    assert idx.shape == (100_000,) and idx.min() >= 0 and idx.max() < 4
    assert np.bincount(idx, minlength=4) / len(idx) == pytest.approx(p, abs=0.005)


def test_nominal_with_many_levels_and_other() -> None:
    rng = np.random.default_rng(0)
    counts = [900, 600, 400, 300, 250, 200, 150, 100, 50, 30, 8, 7, 5]
    values = np.repeat([f"h{i}" for i in range(len(counts))], counts)
    df = pd.DataFrame({"hospital": rng.permutation(values), "x": rng.normal(size=len(values))})
    p = _quiet_profile(df, types={"hospital": "nominal"}, pseudonymize=["hospital"])
    col = p.column("hospital")
    assert col.other_level == "Other"
    g = generate(p, n=20000, seed=5)["hospital"]
    assert g.notna().all() and set(g) <= set(col.levels)
    got = g.value_counts(normalize=True).reindex(col.levels, fill_value=0).to_numpy()
    assert got == pytest.approx(col.probabilities, abs=0.01)


def test_categorical_source_dtype_is_restored() -> None:
    rng = np.random.default_rng(1)
    stage = pd.Categorical(rng.choice(["I", "II", "III"], size=1000),
                           categories=["I", "II", "III"], ordered=True)
    g = generate(_quiet_profile(pd.DataFrame({"stage": stage})), seed=0)
    assert isinstance(g["stage"].dtype, pd.CategoricalDtype) and g["stage"].cat.ordered
    assert list(g["stage"].cat.categories) == ["I", "II", "III"]


def test_ids_are_regenerated(clinical_df, generated) -> None:
    ids = generated["patient_id"]
    assert ids.is_unique and ids.iloc[0] == "ID000001"
    assert not set(ids) & set(clinical_df["patient_id"])


def test_constant_and_empty_columns(generated) -> None:
    assert (generated["site"] == "A").all()
    assert generated["unused"].isna().all()


def test_non_positive_definite_matrix_falls_back(clinical_profile) -> None:
    d = clinical_profile.to_dict()
    labels = [lb.key for lb in clinical_profile.latent_labels()]
    i, j, k = labels.index("age"), labels.index("bmi"), labels.index("crp")
    m = d["dependence"]["latent"]["matrix"]
    for a, b, v in ((i, j, 0.95), (i, k, 0.95), (j, k, -0.95)):
        m[a][b] = m[b][a] = v
    broken = Profile.from_dict(d)
    with pytest.warns(KageFrameWarning, match="not positive definite"):
        out = generate(broken, n=500, seed=0)
    assert len(out) == 500 and out["age"].notna().any()


def test_generated_correlation_roundtrip(clinical_profile) -> None:
    dummy = generate(clinical_profile, seed=7, missing="none")
    again = _quiet_profile(dummy, types=RECOMMENDED_TYPES)
    labels = clinical_profile.latent_labels()
    assert again.latent_labels() == labels
    plain = np.array([lb.level is None for lb in labels])
    a = np.array(clinical_profile.dependence.latent.matrix, dtype=float)[np.ix_(plain, plain)]
    b = np.array(again.dependence.latent.matrix, dtype=float)[np.ix_(plain, plain)]
    assert np.max(np.abs(a - b)) < 0.07


def test_example_profile_with_all_types_generates(example_profile) -> None:
    g = generate(example_profile, seed=0)
    assert list(g.columns) == example_profile.output_columns
    assert isinstance(example_profile.column("sex"), CategoricalColumn)
    assert set(g["sex"].dropna()) <= {"L01", "L02"}
    assert set(generate(example_profile, seed=0, level_map=example_profile.level_map)["sex"]
               .dropna()) <= {"M", "F"}
    assert example_profile.column("admit_dt").type is T.DATETIME
    assert str(g["admit_dt"].dtype).endswith("Asia/Tokyo]")


def test_invalid_arguments(clinical_profile) -> None:
    with pytest.raises(ValueError):
        generate(clinical_profile, n=0)
    with pytest.raises(ValueError, match="missing"):
        generate(clinical_profile, missing="sometimes")


def test_profile_without_latent_columns() -> None:
    df = pd.DataFrame({"id": [f"P{i:05d}" for i in range(200)], "site": "A"})
    g = generate(_quiet_profile(df), seed=0)
    assert list(g.columns) == ["id", "site"] and g["id"].is_unique
