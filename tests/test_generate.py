"""generate(): reconstruction of each column type from a profile.

The MVP acceptance criteria (columns, marginals, frequencies, missing rates, dependence,
nominal, reproducibility) are tested end to end in ``test_mvp_acceptance.py``; this file
covers the mechanics of generation.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from kageframe import generate, profile_dataframe
from kageframe.datasets import RECOMMENDED_TYPES
from kageframe.exceptions import KageFrameWarning
from kageframe.latent import nominal_choice
from kageframe.schema import CategoricalColumn, Profile
from kageframe.types import ColumnType as T


@pytest.fixture(scope="module")
def generated(clinical_profile) -> pl.DataFrame:
    return generate(clinical_profile, seed=123)


def _quiet_profile(df: pl.DataFrame, **kwargs) -> Profile:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return profile_dataframe(df, **kwargs)


def test_default_n_and_explicit_n(clinical_df, clinical_profile, generated) -> None:
    assert len(generated) == len(clinical_df)
    assert len(generate(clinical_profile, n=250, seed=0)) == 250


def test_dtypes_follow_the_source(clinical_df, generated) -> None:
    for name in ("age", "smoker", "ecog"):
        assert generated[name].dtype == pl.Int64
        assert generated[name].null_count() > 0
    assert generated["bmi"].dtype == pl.Float64
    assert generated["lab_flag"].dtype == pl.Boolean
    assert generated["exam_date"].dtype == clinical_df["exam_date"].dtype
    assert generated["admit_dt"].dtype == clinical_df["admit_dt"].dtype
    assert set(generated["sex"].drop_nulls()) <= {"F", "M"}


def test_integer_and_float_widths_are_restored() -> None:
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.integers(0, 1000, 2000), "y": rng.normal(size=2000)})
    df = df.with_columns(pl.col("x").cast(pl.Int32), pl.col("y").cast(pl.Float32))
    out = generate(_quiet_profile(df, types={"x": "numeric"}), seed=0)
    assert out.schema == df.schema


def test_integer_valued_float_column_stays_float() -> None:
    df = pl.DataFrame({"x": np.random.default_rng(0).integers(0, 1000, 2000).astype(float)})
    out = generate(_quiet_profile(df, types={"x": "numeric"}), seed=0)
    assert out["x"].dtype == pl.Float64


def test_numeric_values_stay_inside_blurred_range(clinical_profile, generated) -> None:
    for name in ("age", "bmi", "crp"):
        col = clinical_profile.column(name)
        g = generated[name].drop_nulls().cast(pl.Float64).to_numpy()
        assert g.min() >= col.min and g.max() <= col.max
        assert np.all(np.round(g, col.decimals) == g)


def test_discrete_numeric_support() -> None:
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"pain": rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1], size=5000)})
    p = profile_dataframe(df, types={"pain": "numeric"})
    freq = generate(p, seed=0)["pain"].value_counts(normalize=True).sort("pain")
    assert freq["pain"].to_list() == [0, 1, 2, 3]
    assert freq["proportion"].to_numpy() == pytest.approx(p.column("pain").probabilities,
                                                          abs=0.015)


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
    df = pl.DataFrame({"hospital": rng.permutation(values).tolist(),
                       "x": rng.normal(size=len(values))})
    p = _quiet_profile(df, types={"hospital": "nominal"}, pseudonymize=["hospital"])
    col = p.column("hospital")
    assert col.other_level == "Other"
    g = generate(p, n=20000, seed=5)["hospital"]
    assert g.null_count() == 0 and set(g) <= set(col.levels)
    share = dict(g.value_counts(normalize=True).iter_rows())
    got = np.array([share.get(lv, 0.0) for lv in col.levels])
    assert got == pytest.approx(col.probabilities, abs=0.01)


def test_enum_and_categorical_source_dtypes_are_restored() -> None:
    rng = np.random.default_rng(1)
    df = pl.DataFrame({
        "stage": pl.Series(rng.choice(["I", "II", "III"], size=1000).tolist(),
                           dtype=pl.Enum(["I", "II", "III"])),
        "site": pl.Series(rng.choice(["a", "b", "c"], size=1000).tolist(),
                          dtype=pl.Categorical),
    })
    p = _quiet_profile(df)
    assert p.column("stage").source_dtype == "Enum"
    g = generate(p, seed=0)
    assert g["stage"].dtype == pl.Enum(["I", "II", "III"])
    assert g["site"].dtype == pl.Categorical()


def test_enum_with_pseudonymized_levels() -> None:
    rng = np.random.default_rng(1)
    df = pl.DataFrame({"stage": pl.Series(rng.choice(["I", "II", "III"], size=1000).tolist(),
                                          dtype=pl.Enum(["I", "II", "III"]))})
    p = _quiet_profile(df, pseudonymize=["stage"])
    assert "I" not in p.to_json()
    assert generate(p, seed=0)["stage"].dtype == pl.Enum(["L01", "L02", "L03"])
    real = generate(p, seed=0, level_map=p.level_map)["stage"]
    assert real.dtype == pl.Enum(["I", "II", "III"])


def test_ids_are_regenerated(clinical_df, generated) -> None:
    ids = generated["patient_id"]
    assert ids.is_unique().all() and ids[0] == "ID000001"
    assert not set(ids) & set(clinical_df["patient_id"])


def test_constant_and_empty_columns(generated) -> None:
    assert (generated["site"] == "A").all()
    assert generated["unused"].is_null().all()


def test_non_positive_definite_matrix_falls_back(clinical_profile) -> None:
    d = clinical_profile.to_dict()
    labels = [lb.key for lb in clinical_profile.latent_labels()]
    i, j, k = labels.index("age"), labels.index("bmi"), labels.index("crp")
    pairs = {(p[0], p[1]): p for p in d["dependence"]["pairs"]}
    for a, b, v in ((i, j, 0.95), (i, k, 0.95), (j, k, -0.95)):
        a, b = sorted((a, b))
        pairs[(a, b)] = [a, b, v, v]
    d["dependence"]["pairs"] = [pairs[key] for key in sorted(pairs)]
    d["dependence"]["psd_correction"]["applied"] = False
    broken = Profile.from_dict(d)
    with pytest.warns(KageFrameWarning, match="not positive definite"):
        out = generate(broken, n=500, seed=0)
    assert len(out) == 500 and out["age"].is_not_null().any()


def test_generated_correlation_roundtrip(clinical_profile) -> None:
    dummy = generate(clinical_profile, seed=7, missing="none")
    again = _quiet_profile(dummy, types=RECOMMENDED_TYPES)
    labels = clinical_profile.latent_labels()
    assert again.latent_labels() == labels
    plain = np.array([lb.level is None for lb in labels])
    a = clinical_profile.dependence.latent_matrix()[np.ix_(plain, plain)]
    b = again.dependence.latent_matrix()[np.ix_(plain, plain)]
    assert np.max(np.abs(a - b)) < 0.07


def test_example_profile_with_all_types_generates(example_profile) -> None:
    g = generate(example_profile, seed=0)
    assert list(g.columns) == example_profile.output_columns
    assert isinstance(example_profile.column("sex"), CategoricalColumn)
    assert set(g["sex"].drop_nulls()) <= {"L01", "L02"}
    assert set(generate(example_profile, seed=0, level_map=example_profile.level_map)["sex"]
               .drop_nulls()) <= {"M", "F"}
    assert example_profile.column("admit_dt").type is T.DATETIME
    assert g["admit_dt"].dtype == pl.Datetime("us", "Asia/Tokyo")


def test_invalid_arguments(clinical_profile) -> None:
    with pytest.raises(ValueError):
        generate(clinical_profile, n=0)
    with pytest.raises(ValueError, match="missing"):
        generate(clinical_profile, missing="sometimes")


def test_profile_without_latent_columns() -> None:
    df = pl.DataFrame({"id": [f"P{i:05d}" for i in range(200)], "site": ["A"] * 200})
    g = generate(_quiet_profile(df), seed=0)
    assert list(g.columns) == ["id", "site"] and g["id"].is_unique().all()


def test_id_values_replace_sequential_ids(clinical_profile) -> None:
    keys = [f"K{i}" for i in range(300)]
    g = generate(clinical_profile, n=300, seed=0, id_values={"patient_id": keys})
    assert g["patient_id"].to_list() == keys
    with pytest.raises(ValueError, match="expected n=300"):
        generate(clinical_profile, n=300, seed=0, id_values={"patient_id": keys[:10]})
    with pytest.raises(ValueError, match="not id columns"):
        generate(clinical_profile, n=300, seed=0, id_values={"age": keys})
