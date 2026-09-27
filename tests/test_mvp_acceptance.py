"""MVP acceptance tests: the seven completion criteria of KageFrame V0.1.

Every test follows the intended workflow on the synthetic clinical fixture
(``make_clinical_like_df(n=10000, seed=0)``, which contains every column type)::

    profile = profile_dataframe(df, types=RECOMMENDED_TYPES)
    dummy = generate(profile, n=len(df), seed=42)
    report = compare(profile, dummy)

Tolerances are those of the plan (§7): about three sampling standard errors at n=10,000
plus the approximation error of the method.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from kageframe import compare, generate, load_profile, profile_dataframe
from kageframe.datasets import RECOMMENDED_TYPES, make_clinical_like_df

SEED = 42
FREE_TEXT = ["note"]
CONTINUOUS = ["age", "bmi", "crp", "exam_date", "admit_dt", "visit_time"]
CATEGORICAL = ["sex", "smoker", "stage", "ecog", "indication", "lab_flag"]


@pytest.fixture(scope="module")
def real() -> pd.DataFrame:
    return make_clinical_like_df(n=10000, seed=0)


@pytest.fixture(scope="module")
def profile(real):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return profile_dataframe(real, types=RECOMMENDED_TYPES)


@pytest.fixture(scope="module")
def dummy(real, profile) -> pd.DataFrame:
    return generate(profile, n=len(real), seed=SEED)


@pytest.fixture(scope="module")
def report(profile, dummy):
    return compare(profile, dummy)


def test_1_same_columns(real, dummy) -> None:
    """Same columns in the same order (free text is dropped on purpose), same row count."""
    assert list(dummy.columns) == [c for c in real.columns if c not in FREE_TEXT]
    assert len(dummy) == len(real)
    for name in ("age", "bmi", "crp"):
        assert pd.api.types.is_numeric_dtype(dummy[name])
    for name in ("exam_date", "admit_dt"):
        assert pd.api.types.is_datetime64_any_dtype(dummy[name])
    assert dummy["visit_time"].dropna().str.fullmatch(r"\d{2}:\d{2}").all()
    # identifiers are regenerated and never reuse a real id
    assert dummy["patient_id"].is_unique
    assert not set(dummy["patient_id"]) & set(real["patient_id"])


@pytest.mark.parametrize("name", CONTINUOUS)
def test_2_marginals_roughly_match(report, name) -> None:
    """Numeric, date and time distributions: CDF distance on the profile grid <= 0.02;
    numeric mean within 0.05 SD and SD within 5% (IQR within 10% for heavy tails)."""
    row = report.summary().loc[name]
    assert row["grid_ks"] <= 0.02
    if name == "admit_dt":
        assert row["grid_ks_time"] <= 0.02
    if name in ("age", "bmi", "crp"):
        assert abs(row["mean_diff_sd"]) <= 0.05
        if row["heavy_tailed"]:
            assert abs(row["iqr_rel_diff"]) <= 0.10
        else:
            assert abs(row["sd_rel_diff"]) <= 0.05


@pytest.mark.parametrize("name", CATEGORICAL)
def test_3_category_frequencies_roughly_match(report, name) -> None:
    """Per-level frequency difference <= 0.015, TVD <= 0.03, and no level outside the profile."""
    row = report.summary().loc[name]
    assert row["max_abs_dp"] <= 0.015
    assert row["tvd"] <= 0.03
    assert row["unseen_values"] == 0


def test_4_missing_rates_match(real, profile, dummy) -> None:
    """Exact mode reproduces every missing rate up to rounding; Bernoulli mode within 3 SE."""
    n = len(dummy)
    for name in dummy.columns:
        assert abs(dummy[name].isna().mean() - real[name].isna().mean()) <= 1 / n, name
    bern = generate(profile, n=n, seed=SEED, missing="bernoulli")
    for name in dummy.columns:
        p = real[name].isna().mean()
        assert abs(bern[name].isna().mean() - p) <= 3 * np.sqrt(p * (1 - p) / n) + 0.001, name


def test_5_pairwise_dependence_roughly_matches(report) -> None:
    """The dummy's latent correlations, estimated the same way, match the profile's:
    <= 0.03 between continuous dimensions, <= 0.05 with binary/ordinal, <= 0.07 with nominal
    levels, RMSE <= 0.02. (Recovery of the true correlation is tested in test_dependence.)"""
    table = report.correlation_table()
    worst = table.groupby("kind")["abs_diff"].max()
    assert worst["continuous"] <= 0.03
    assert worst["discrete"] <= 0.05
    assert worst["nominal"] <= 0.07
    assert report.correlation_summary()["rmse"] <= 0.02


def test_6_nominal_has_exactly_one_level(profile, dummy) -> None:
    """Every observed row of a nominal column holds exactly one known level."""
    levels = profile.column("indication").levels
    values = dummy["indication"]
    observed = values.notna()
    assert set(values[observed]) <= set(levels)
    one_hot = pd.get_dummies(values).reindex(columns=levels, fill_value=False)
    row_sums = one_hot.sum(axis=1)
    assert (row_sums[observed] == 1).all()
    assert (row_sums[~observed] == 0).all()


def test_7_seed_reproducibility(tmp_path, real, profile, dummy) -> None:
    """Same seed -> identical data (also after save/load); another seed -> different data;
    profiling the same DataFrame twice gives the identical JSON."""
    pd.testing.assert_frame_equal(generate(profile, n=len(real), seed=SEED), dummy)
    assert not generate(profile, n=len(real), seed=SEED + 1).equals(dummy)
    profile.save(tmp_path / "profile.json")
    pd.testing.assert_frame_equal(
        generate(load_profile(tmp_path / "profile.json"), n=len(real), seed=SEED), dummy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = profile_dataframe(real, types=RECOMMENDED_TYPES)
    assert again.to_json() == profile.to_json()


def test_all_default_checks_pass(report) -> None:
    """``CompareReport.check()`` bundles criteria 1-5 with the default tolerances."""
    result = report.check()
    assert result, str(result)


@pytest.mark.slow
def test_checks_pass_for_many_seeds(profile) -> None:
    """The tolerances do not depend on a lucky seed: 20 seeds all pass."""
    failures = {seed: compare(profile, generate(profile, seed=seed)).check().failures
                for seed in range(20)}
    assert not {s: f for s, f in failures.items() if f}
