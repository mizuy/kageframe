"""Missing values: generate() reproduces each column's missing rate (MCAR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kageframe import generate, load_profile, profile_dataframe
from kageframe.generate import missing_mask


def test_exact_mode_reproduces_missing_rates(clinical_profile) -> None:
    dummy = generate(clinical_profile, seed=3)  # missing="exact" is the default
    for col in clinical_profile.columns:
        if col.in_output:
            assert dummy[col.name].isna().sum() == round(len(dummy) * col.missing_rate), col.name


def test_exact_mode_scales_with_n(clinical_profile) -> None:
    dummy = generate(clinical_profile, n=1234, seed=3)
    crp = clinical_profile.column("crp")
    assert dummy["crp"].isna().sum() == round(1234 * crp.missing_rate)


def test_bernoulli_mode_is_close_to_the_rate(clinical_profile) -> None:
    dummy = generate(clinical_profile, n=20000, seed=3, missing="bernoulli")
    for name in ("bmi", "crp", "stage", "visit_time", "lab_flag"):
        p = clinical_profile.column(name).missing_rate
        se = np.sqrt(p * (1 - p) / len(dummy))
        assert abs(dummy[name].isna().mean() - p) < 4 * se + 1e-3, name


def test_none_mode_has_no_missing_values(clinical_profile) -> None:
    dummy = generate(clinical_profile, seed=3, missing="none")
    assert dummy.drop(columns=["unused"]).notna().all().all()
    assert dummy["unused"].isna().all()  # an all-missing column stays all missing


def test_missingness_does_not_change_the_other_values(clinical_profile) -> None:
    """The mask uses its own random stream, so observed values are the same in every mode."""
    full = generate(clinical_profile, seed=11, missing="none")
    masked = generate(clinical_profile, seed=11)
    for name in ("age", "indication", "exam_date", "admit_dt", "visit_time"):
        keep = masked[name].notna()
        assert (masked.loc[keep, name].astype(object) == full.loc[keep, name].astype(object)).all()


def test_masks_are_independent_between_columns(clinical_profile) -> None:
    dummy = generate(clinical_profile, n=20000, seed=5)
    a, b = dummy["crp"].isna(), dummy["visit_time"].isna()
    expected = a.mean() * b.mean()
    assert (a & b).mean() == pytest.approx(expected, abs=0.006)


def test_missing_is_completely_at_random(clinical_profile) -> None:
    """Rows hidden in one column look like the others in every other column (MCAR)."""
    dummy = generate(clinical_profile, n=20000, seed=8)
    hidden = dummy["crp"].isna()
    age = dummy["age"].astype(float)
    assert abs(age[hidden].mean() - age[~hidden].mean()) < 0.1 * age.std()


def test_datetime_parts_are_missing_together(clinical_profile) -> None:
    dummy = generate(clinical_profile, seed=2)
    admit = dummy["admit_dt"]
    assert admit.isna().sum() == round(len(dummy) * clinical_profile.column("admit_dt")
                                       .missing_rate)
    assert admit.dropna().dt.hour.notna().all()


def test_missing_values_use_nullable_dtypes(clinical_profile) -> None:
    dummy = generate(clinical_profile, seed=1)
    assert dummy["age"].dtype == "Int64" and dummy["age"].isna().any()
    assert dummy["lab_flag"].dtype == "boolean" and dummy["lab_flag"].isna().any()
    assert dummy["bmi"].isna().any() and dummy["bmi"].dtype == np.float64
    assert dummy["exam_date"].isna().any() and str(dummy["exam_date"].dtype).startswith(
        "datetime64")
    assert dummy["sex"].notna().all()  # no missing values in the source column


def test_missing_mask_exact_and_seeded() -> None:
    rng = np.random.default_rng(0)
    mask = missing_mask(1000, 0.123, "exact", rng)
    assert mask.sum() == 123
    again = missing_mask(1000, 0.123, "exact", np.random.default_rng(0))
    assert np.array_equal(mask, again)
    assert not missing_mask(1000, 0.0, "exact", rng).any()
    assert missing_mask(1000, 1.0, "exact", rng).all()


def test_missing_rate_of_small_source_is_kept(tmp_path) -> None:
    df = pd.DataFrame({"x": [1.5, None, 2.5, 3.5] * 50})
    p = profile_dataframe(df)
    p.save(tmp_path / "p.json")
    dummy = generate(load_profile(tmp_path / "p.json"), seed=0)
    assert dummy["x"].isna().mean() == pytest.approx(0.25)
