"""compare(): marginal and latent-correlation differences between a profile and a dummy.

A dummy generated from the profile itself must pass ``check()``; dummies with injected
faults (a dropped column, a shifted distribution, a new level, a changed missing rate,
broken dependence) must fail it with a message naming the problem.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from kageframe import DEFAULT_TOLERANCES, CompareReport, compare, generate, profile_dataframe
from kageframe.compare import grid_ks
from kageframe.datasets import RECOMMENDED_TYPES
from kageframe.schema import Quantiles


@pytest.fixture(scope="module")
def dummy(clinical_profile) -> pd.DataFrame:
    return generate(clinical_profile, seed=42)


@pytest.fixture(scope="module")
def report(clinical_profile, dummy) -> CompareReport:
    return compare(clinical_profile, dummy)


def _failures(profile, df) -> list[str]:
    return compare(profile, df).check().failures


# --- a faithful dummy passes ----------------------------------------------------------------


def test_dummy_from_the_profile_passes(report) -> None:
    result = report.check()
    assert result, str(result)
    assert report.columns_match


def test_summary_has_one_row_per_output_column(clinical_profile, report) -> None:
    summary = report.summary()
    assert list(summary.index) == clinical_profile.output_columns
    for key in ("missing_rate_profile", "missing_rate_dummy", "grid_ks", "mean_diff_sd",
                "max_abs_dp", "tvd", "max_corr_diff"):
        assert key in summary.columns
    assert summary.loc["age", "missing_rate_dummy"] == pytest.approx(0.01)
    assert summary.loc["crp", "heavy_tailed"]


def test_correlation_table_reports_latent_differences(clinical_profile, report) -> None:
    table = report.correlation_table()
    assert {"a", "b", "kind", "profile", "dummy", "abs_diff", "profile_latent",
            "dummy_latent", "latent_abs_diff"} <= set(table.columns)
    assert set(table["kind"]) == {"continuous", "discrete", "nominal"}
    row = table[(table.a == "age") & (table.b == "ecog")].iloc[0]
    assert row.abs_diff == pytest.approx(abs(row.dummy - row.profile))
    summary = report.correlation_summary()
    assert summary["max_abs_diff"] < 0.07 and summary["rmse"] < 0.02


def test_report_is_json_serializable(report) -> None:
    d = json.loads(json.dumps(report.to_dict()))
    assert d["columns"]["match"] is True
    assert len(d["marginals"]) == len(report.marginals)


def test_level_map_lets_real_names_be_compared(clinical_df) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = profile_dataframe(clinical_df, types=RECOMMENDED_TYPES, pseudonymize=["indication"])
    real_names = generate(p, seed=1, level_map=p.level_map)
    assert not compare(p, real_names).check()  # real names look like unseen levels
    assert compare(p, real_names, level_map=p.level_map).check()


# --- injected faults are detected ------------------------------------------------------------


def test_missing_extra_and_reordered_columns(clinical_profile, dummy) -> None:
    fails = _failures(clinical_profile, dummy.drop(columns=["bmi"]))
    assert any("missing ['bmi']" in f for f in fails)
    fails = _failures(clinical_profile, dummy.assign(extra=1))
    assert any("extra ['extra']" in f for f in fails)
    reordered = dummy[list(reversed(dummy.columns))]
    assert "column order differs" in _failures(clinical_profile, reordered)


def test_shifted_numeric_distribution(clinical_profile, dummy) -> None:
    shifted = dummy.assign(age=dummy["age"] + 5)
    fails = _failures(clinical_profile, shifted)
    assert any(f.startswith("age: mean_diff_sd") for f in fails)
    assert any(f.startswith("age: grid_ks") for f in fails)


def test_wider_numeric_distribution(clinical_profile, dummy) -> None:
    bmi = dummy["bmi"]
    wider = dummy.assign(bmi=bmi.mean() + 1.3 * (bmi - bmi.mean()))
    assert any(f.startswith("bmi: sd_rel_diff") for f in _failures(clinical_profile, wider))


def test_shifted_dates(clinical_profile, dummy) -> None:
    shifted = dummy.assign(exam_date=dummy["exam_date"] + pd.Timedelta(days=90))
    assert any(f.startswith("exam_date: grid_ks") for f in _failures(clinical_profile, shifted))


def test_shifted_times(clinical_profile, dummy) -> None:
    later = dummy.assign(admit_dt=dummy["admit_dt"] + pd.Timedelta(hours=3))
    report = compare(clinical_profile, later)
    assert report.summary().loc["admit_dt", "grid_ks_time"] > 0.05


def test_changed_category_frequencies(clinical_profile, dummy) -> None:
    sex = dummy["sex"].copy()
    sex.iloc[: len(sex) // 5] = "M"
    fails = _failures(clinical_profile, dummy.assign(sex=sex))
    assert any(f.startswith("sex: max_abs_dp") for f in fails)


def test_unseen_level(clinical_profile, dummy) -> None:
    indication = dummy["indication"].astype(object).copy()
    indication.iloc[:5] = "brand_new"
    fails = _failures(clinical_profile, dummy.assign(indication=indication))
    assert "indication: 5 values not in the profile" in fails


def test_changed_missing_rate(clinical_profile, dummy) -> None:
    crp = dummy["crp"].copy()
    crp.iloc[:1000] = np.nan
    fails = _failures(clinical_profile, dummy.assign(crp=crp))
    assert any(f.startswith("crp: missing rate differs") for f in fails)
    assert not generate(clinical_profile, seed=1, missing="none").pipe(
        lambda d: compare(clinical_profile, d).check())


def test_broken_dependence(clinical_profile, dummy) -> None:
    rng = np.random.default_rng(0)
    shuffled = dummy.assign(ecog=dummy["ecog"].to_numpy()[rng.permutation(len(dummy))])
    report = compare(clinical_profile, shuffled)
    fails = report.check().failures
    assert any("correlation age x ecog" in f for f in fails)
    table = report.correlation_table().set_index(["a", "b"])
    assert table.loc[("age", "ecog"), "abs_diff"] > 0.2


def test_constant_and_id_columns(clinical_profile, dummy) -> None:
    fails = _failures(clinical_profile, dummy.assign(site=["A", "B"] * (len(dummy) // 2)))
    assert "site: constant column has other values" in fails
    fails = _failures(clinical_profile, dummy.assign(patient_id="ID000001"))
    assert "patient_id: identifiers are not unique" in fails


# --- tolerances --------------------------------------------------------------------------------


def test_custom_tolerances(report) -> None:
    assert not report.check(tol={"corr_rmse": 1e-6, "corr_rmse_se": 0.0})
    strict = report.check(tol={"corr_continuous": 0.001, "corr_z": 0.0})
    assert any(f.startswith("correlation age x bmi") for f in strict.failures)
    assert report.check(tol={"corr_rmse": 1.0})
    with pytest.raises(ValueError, match="unknown tolerances"):
        report.check(tol={"nonsense": 1.0})
    assert DEFAULT_TOLERANCES["grid_ks"] == 0.02


def test_small_dummies_get_wider_tolerances(clinical_profile) -> None:
    small = generate(clinical_profile, n=1000, seed=3)
    assert compare(clinical_profile, small).check()


def test_grid_ks_handles_ties() -> None:
    grid = Quantiles([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0, 1.0, 2.0])
    same = np.array([0.0] * 60 + [1.0] * 20 + [2.0] * 20)
    assert grid_ks(same, grid) == 0.0  # every p falls inside the jump at its tied value
    x = np.array([0.0] * 50 + [1.0] * 20 + [2.0] * 30)
    assert grid_ks(x, grid) == pytest.approx(0.05)  # F(1) = 0.7 < p = 0.75
