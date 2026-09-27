from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from clinmock import generate, profile_dataframe
from clinmock.datasets import RECOMMENDED_TYPES, make_clinical_like_df
from clinmock.exceptions import ClinmockWarning
from clinmock.schema import CategoricalColumn, Profile, load_profile
from clinmock.types import ColumnType as T

CATEGORICAL = ("sex", "smoker", "stage", "ecog", "indication", "lab_flag")


@pytest.fixture(scope="module")
def generated(clinical_profile) -> pd.DataFrame:
    return generate(clinical_profile, seed=123)


def _ecdf(x: np.ndarray, at: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(x), at, side="right") / len(x)


# --- MVP: columns --------------------------------------------------------------------------


def test_output_columns(clinical_df, clinical_profile, generated) -> None:
    dropped = {"note", "exam_date", "admit_dt", "visit_time"}
    expected = [c for c in clinical_df.columns if c not in dropped]
    assert list(generated.columns) == expected == clinical_profile.output_columns
    assert len(generated) == len(clinical_df)
    assert len(generate(clinical_profile, n=250, seed=0)) == 250


def test_output_dtypes(generated) -> None:
    assert generated["age"].dtype == np.int64
    assert generated["bmi"].dtype == np.float64
    assert generated["smoker"].dtype == np.int64 and generated["ecog"].dtype == np.int64
    assert generated["lab_flag"].dtype == bool
    assert set(generated["sex"]) <= {"F", "M"}


# --- MVP: marginal distributions -----------------------------------------------------------


@pytest.mark.parametrize("name", ["age", "bmi"])
def test_numeric_mean_and_sd(clinical_df, generated, name) -> None:
    x = clinical_df[name].dropna().to_numpy(dtype=float)
    g = generated[name].to_numpy(dtype=float)
    assert abs(g.mean() - x.mean()) < 0.05 * x.std()
    assert g.std() == pytest.approx(x.std(), rel=0.05)


@pytest.mark.parametrize("name", ["age", "bmi", "crp"])
def test_numeric_distribution_close_on_grid(clinical_df, clinical_profile, generated,
                                            name) -> None:
    x = clinical_df[name].dropna().to_numpy(dtype=float)
    g = generated[name].to_numpy(dtype=float)
    at = np.array(clinical_profile.column(name).grid.values)
    assert np.max(np.abs(_ecdf(g, at) - _ecdf(x, at))) <= 0.02
    col = clinical_profile.column(name)
    assert g.min() >= col.min and g.max() <= col.max
    assert np.all(np.round(g, col.decimals) == g)


def test_heavy_tailed_iqr(clinical_df, generated) -> None:
    x = clinical_df["crp"].dropna().to_numpy(dtype=float)
    g = generated["crp"].to_numpy(dtype=float)
    iqr = np.subtract(*np.quantile(x, [0.75, 0.25]))
    assert np.subtract(*np.quantile(g, [0.75, 0.25])) == pytest.approx(iqr, rel=0.1)
    assert np.mean(g == 0) == pytest.approx(np.mean(x == 0), abs=0.015)


def test_discrete_numeric_support() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"pain": rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1], size=5000)})
    p = profile_dataframe(df, types={"pain": "numeric"})
    g = generate(p, seed=0)["pain"]
    freq = g.value_counts(normalize=True).sort_index()
    assert list(freq.index) == [0, 1, 2, 3]
    assert freq.to_numpy() == pytest.approx(p.column("pain").probabilities, abs=0.015)


# --- MVP: category frequencies and nominal ---------------------------------------------------


@pytest.mark.parametrize("name", CATEGORICAL)
def test_category_frequencies(clinical_profile, generated, name) -> None:
    col = clinical_profile.column(name)
    assert isinstance(col, CategoricalColumn)
    freq = generated[name].value_counts(normalize=True)
    assert set(freq.index) <= set(col.levels)
    got = np.array([freq.get(lv, 0.0) for lv in col.levels])
    assert got == pytest.approx(col.probabilities, abs=0.015)


def test_nominal_has_exactly_one_level_per_row(clinical_profile, generated) -> None:
    col = clinical_profile.column("indication")
    values = generated["indication"]
    assert values.notna().all()
    indicators = np.column_stack([(values == lv).to_numpy() for lv in col.levels])
    assert np.all(indicators.sum(axis=1) == 1)


def test_nominal_with_many_levels_and_other() -> None:
    rng = np.random.default_rng(0)
    counts = [900, 600, 400, 300, 250, 200, 150, 100, 50, 30, 8, 7, 5]
    values = np.repeat([f"h{i}" for i in range(len(counts))], counts)
    df = pd.DataFrame({"hospital": rng.permutation(values), "x": rng.normal(size=len(values))})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = profile_dataframe(df, types={"hospital": "nominal"}, pseudonymize=["hospital"])
    col = p.column("hospital")
    assert col.other_level == "Other"
    g = generate(p, n=20000, seed=5)["hospital"]
    assert g.notna().all() and set(g) <= set(col.levels)
    got = g.value_counts(normalize=True).reindex(col.levels, fill_value=0).to_numpy()
    assert got == pytest.approx(col.probabilities, abs=0.01)


# --- MVP: seed reproducibility -------------------------------------------------------------


def test_same_seed_same_output(clinical_profile, generated) -> None:
    pd.testing.assert_frame_equal(generate(clinical_profile, seed=123), generated)


def test_different_seed_differs(clinical_profile, generated) -> None:
    other = generate(clinical_profile, seed=124)
    assert not other.equals(generated)
    assert (other["age"] != generated["age"]).mean() > 0.5


def test_saved_profile_reproduces_output(tmp_path, clinical_profile, generated) -> None:
    path = tmp_path / "profile.json"
    clinical_profile.save(path)
    loaded = load_profile(path)
    assert loaded.to_json() == clinical_profile.to_json()
    pd.testing.assert_frame_equal(generate(loaded, seed=123), generated)


def test_profile_from_same_data_and_seed_reproduces_output() -> None:
    frames = []
    for _ in range(2):
        df = make_clinical_like_df(2000, seed=9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frames.append(generate(profile_dataframe(df, types=RECOMMENDED_TYPES), seed=1))
    pd.testing.assert_frame_equal(*frames)


# --- other columns and dependence ----------------------------------------------------------


def test_ids_are_regenerated(clinical_df, generated) -> None:
    ids = generated["patient_id"]
    assert ids.is_unique and ids.iloc[0] == "ID000001"
    assert not set(ids) & set(clinical_df["patient_id"])


def test_constant_and_empty_columns(generated) -> None:
    assert (generated["site"] == "A").all()
    assert generated["unused"].isna().all()


def test_generated_correlation_roundtrip(clinical_profile, generated) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = profile_dataframe(generated, types=RECOMMENDED_TYPES)
    labels = clinical_profile.latent_labels()
    assert again.latent_labels() == labels
    plain = np.array([lb.level is None for lb in labels])
    a = np.array(clinical_profile.dependence.latent.matrix, dtype=float)[np.ix_(plain, plain)]
    b = np.array(again.dependence.latent.matrix, dtype=float)[np.ix_(plain, plain)]
    assert np.max(np.abs(a - b)) < 0.07


def test_non_positive_definite_matrix_falls_back(clinical_profile) -> None:
    d = clinical_profile.to_dict()
    labels = [lb.key for lb in clinical_profile.latent_labels()]
    i, j, k = labels.index("age"), labels.index("bmi"), labels.index("crp")
    m = d["dependence"]["latent"]["matrix"]
    for a, b, v in ((i, j, 0.95), (i, k, 0.95), (j, k, -0.95)):
        m[a][b] = m[b][a] = v
    broken = Profile.from_dict(d)
    with pytest.warns(ClinmockWarning, match="not positive definite"):
        out = generate(broken, n=500, seed=0)
    assert len(out) == 500 and out["age"].notna().all()


def test_temporal_columns_not_generated_yet(example_profile) -> None:
    assert example_profile.column("exam_date").type is T.DATE
    with pytest.raises(NotImplementedError):
        generate(example_profile, seed=0)


def test_invalid_n(clinical_profile) -> None:
    with pytest.raises(ValueError):
        generate(clinical_profile, n=0)


def test_profile_without_latent_columns() -> None:
    df = pd.DataFrame({"id": [f"P{i:05d}" for i in range(200)], "site": "A"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = generate(profile_dataframe(df), seed=0)
    assert list(g.columns) == ["id", "site"] and g["id"].is_unique
