import numpy as np
import pandas as pd
import pytest

from clinmock.datasets import (
    EXPECTED_EDGE_CASE_TYPES,
    EXPECTED_INFERRED_TYPES,
    INDICATION_LEVELS,
    LATENT_LABELS,
    MISSING_RATES,
    RARE_INDICATIONS,
    latent_correlation,
    make_clinical_like_df,
)


def test_same_seed_is_reproducible():
    a = make_clinical_like_df(n=500, seed=3)
    b = make_clinical_like_df(n=500, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    a = make_clinical_like_df(n=500, seed=3)
    b = make_clinical_like_df(n=500, seed=4)
    assert not a.equals(b)


def test_columns_cover_every_type(clinical_df):
    expected = {**EXPECTED_INFERRED_TYPES, **EXPECTED_EDGE_CASE_TYPES}
    assert list(clinical_df.columns) == list(expected)
    assert len(clinical_df) == 10000


def test_missing_rates_are_exact(clinical_df):
    for name, rate in MISSING_RATES.items():
        assert clinical_df[name].isna().mean() == pytest.approx(rate, abs=1e-4), name


def test_value_domains(clinical_df):
    df = clinical_df
    assert df["patient_id"].is_unique
    assert set(df["sex"].dropna()) == {"M", "F"}
    assert set(df["smoker"].dropna()) == {0, 1}
    assert set(df["stage"].dropna()) == {"I", "II", "III", "IV"}
    assert set(df["ecog"].dropna()) <= {0, 1, 2, 3, 4}
    assert (df["crp"].dropna() == 0).mean() == pytest.approx(0.3, abs=0.03)
    assert (df["bmi"].dropna() * 10 % 1 == 0).all()
    assert (df["exam_date"].dropna().dt.hour == 0).all()
    assert not (df["admit_dt"].dropna().dt.hour == 0).all()
    assert df["visit_time"].dropna().str.fullmatch(r"\d{2}:\d{2}").all()
    assert df["site"].eq("A").all()
    assert df["unused"].isna().all()


def test_indication_frequencies_and_rare_levels(clinical_df):
    counts = clinical_df["indication"].value_counts()
    for level, count in RARE_INDICATIONS.items():
        assert 0 < counts.get(level, 0) <= count
    freqs = clinical_df["indication"].value_counts(normalize=True)
    for level, p in zip(INDICATION_LEVELS, (0.20, 0.42, 0.15, 0.23), strict=True):
        assert freqs[level] == pytest.approx(p, abs=0.02)


def test_latent_correlation_is_valid():
    labels, corr = latent_correlation()
    assert labels == list(LATENT_LABELS)
    np.testing.assert_allclose(np.diag(corr), 1.0)
    np.testing.assert_allclose(corr, corr.T)
    assert np.linalg.eigvalsh(corr).min() > 0.05
    block = [labels.index(f"indication[{lv}]") for lv in INDICATION_LEVELS]
    np.testing.assert_allclose(corr[np.ix_(block, block)], np.eye(len(block)))


def test_latent_correlation_is_reflected_in_data():
    df = make_clinical_like_df(n=20000, seed=1, missing=False)
    rho = df["age"].corr(df["ecog"].astype(float), method="spearman")
    assert rho > 0.2


def test_small_n_rejected():
    with pytest.raises(ValueError):
        make_clinical_like_df(n=10)
