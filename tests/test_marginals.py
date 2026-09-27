from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from kageframe import generate, profile_dataframe
from kageframe.datasets import RARE_INDICATIONS, RECOMMENDED_TYPES
from kageframe.exceptions import KageFrameWarning, PrivacyWarning
from kageframe.marginals.numeric import detect_decimals
from kageframe.schema import (
    ConstantColumn,
    IdColumn,
    LevelMap,
    NumericColumn,
    Profile,
    ReasonColumn,
)
from kageframe.types import ColumnType as T


def quiet_profile(df: pd.DataFrame, **kwargs) -> Profile:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return profile_dataframe(df, **kwargs)


def warning_codes(profile: Profile, column: str | None = None) -> set[str]:
    return {w.code for w in profile.warnings if column is None or w.column == column}


# --- numeric -------------------------------------------------------------------------------


def test_detect_decimals() -> None:
    assert detect_decimals(np.array([1.0, 2.0])) == 0
    assert detect_decimals(np.array([1.5, 2.25])) == 2
    assert detect_decimals(np.array([np.pi])) is None


def test_numeric_quantile_grid_follows_k_rule(clinical_df, clinical_profile) -> None:
    k = clinical_profile.options["min_tail_count"]
    for name in ("age", "bmi", "crp"):
        col = clinical_profile.column(name)
        assert isinstance(col, NumericColumn) and col.mode == "quantile"
        x = np.sort(clinical_df[name].dropna().to_numpy(dtype=float))
        n = len(x)
        probs = np.array(col.grid.probs)
        assert probs[0] == 0.0 and probs[-1] == 1.0
        assert len(probs) <= clinical_profile.options["quantile_grid"] + 6
        # Grid ends are the k/n and 1 - k/n quantiles, never the observed extremes.
        assert col.grid.values[0] == pytest.approx(np.quantile(x, k / n))
        assert col.grid.values[-1] == pytest.approx(np.quantile(x, 1 - k / n))
        assert x[k - 1] <= col.min <= x[k] and x[-k - 1] <= col.max <= x[-k]
        assert np.all(np.diff(col.grid.values) >= 0)


def test_numeric_true_extremes_not_in_profile() -> None:
    rng = np.random.default_rng(3)
    x = np.round(rng.normal(50, 10, 5000), 3)
    x[0], x[1] = -123.456, 987.654
    p = quiet_profile(pd.DataFrame({"x": x}))
    text = p.to_json()
    assert "-123.456" not in text and "987.654" not in text
    col = p.column("x")
    assert col.min > -123.456 and col.max < 987.654


def test_numeric_subtype_and_decimals(clinical_profile) -> None:
    age, bmi, crp = (clinical_profile.column(c) for c in ("age", "bmi", "crp"))
    assert (age.subtype, age.decimals) == ("integer", 0)
    assert (bmi.subtype, bmi.decimals) == ("float", 1)
    assert (crp.subtype, crp.decimals) == ("float", 2)


def test_small_grid_for_small_n() -> None:
    x = np.arange(200, dtype=float) + 0.5
    col = quiet_profile(pd.DataFrame({"x": x})).column("x")
    assert len(col.grid.probs) <= 200 // 10 + 6


def test_numeric_discrete_mode() -> None:
    rng = np.random.default_rng(0)
    x = rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1], size=2000)
    p = quiet_profile(pd.DataFrame({"pain": x}), types={"pain": "numeric"})
    col = p.column("pain")
    assert col.mode == "discrete" and col.support == [0.0, 1.0, 2.0, 3.0]
    assert np.allclose(col.probabilities, np.bincount(x) / len(x))


def test_numeric_rare_value_disables_discrete_mode() -> None:
    x = np.array([0] * 500 + [1] * 500 + [2] * 3)
    col = quiet_profile(pd.DataFrame({"x": x}), types={"x": "numeric"}).column("x")
    assert col.mode == "quantile" and col.support is None


@pytest.mark.filterwarnings("ignore:.*jointly observed")
def test_numeric_too_few_values() -> None:
    x = np.full(100, np.nan)
    x[:15] = np.arange(15) + 0.5
    df = pd.DataFrame({"x": x, "y": np.arange(100) + 0.25})
    with pytest.warns(KageFrameWarning, match="fewer than 20"):
        p = profile_dataframe(df)
    col = p.column("x")
    assert col.grid is None and col.min is None and "too_few_values" in warning_codes(p, "x")
    g = generate(p, seed=0)
    assert g["x"].isna().all() and g["y"].notna().all()


# --- categorical ---------------------------------------------------------------------------


def test_binary_and_ordinal_levels(clinical_df, clinical_profile) -> None:
    sex = clinical_profile.column("sex")
    assert sex.type is T.BINARY and sex.levels == ["F", "M"] and not sex.pseudonymized
    freq = clinical_df["sex"].value_counts(normalize=True)
    assert sex.probabilities == pytest.approx([freq["F"], freq["M"]])
    assert len(sex.thresholds) == 1

    stage = clinical_profile.column("stage")
    assert stage.type is T.ORDINAL and stage.levels == ["I", "II", "III", "IV"]
    assert np.all(np.diff(stage.thresholds) > 0)
    ecog = clinical_profile.column("ecog")
    assert ecog.levels == [0, 1, 2, 3, 4] and ecog.value_kind == "int"
    smoker, lab = clinical_profile.column("smoker"), clinical_profile.column("lab_flag")
    assert smoker.levels == [0, 1] and lab.levels == [False, True]


def test_nominal_frequency_order_and_fold(clinical_df, clinical_profile) -> None:
    col = clinical_profile.column("indication")
    assert col.type is T.NOMINAL and col.other_level is None
    assert col.levels == ["surveillance", "diagnostic", "screening", "FIT_positive"]
    assert np.all(np.diff(col.counts) <= 0)
    assert sum(col.counts) == clinical_df["indication"].notna().sum()
    assert "rare_levels_folded" in warning_codes(clinical_profile, "indication")
    for level in RARE_INDICATIONS:
        assert level not in clinical_profile.to_json()


def _nominal(counts: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame({"v": [lv for lv, c in counts.items() for _ in range(c)]})


def test_rare_nominal_levels_merged_into_other() -> None:
    df = _nominal({"a": 500, "b": 300, "c": 200, "r1": 6, "r2": 5, "r3": 4})
    with pytest.warns(PrivacyWarning, match="merged"):
        p = profile_dataframe(df, types={"v": "nominal"})
    col = p.column("v")
    assert col.levels == ["a", "b", "c", "Other"] and col.counts == [500, 300, 200, 15]
    assert col.other_level == "Other" and col.n_merged_levels == 3
    assert not any(r in p.to_json() for r in ("r1", "r2", "r3"))


def test_rare_merge_joins_existing_other_and_avoids_case_clash() -> None:
    p = quiet_profile(_nominal({"a": 500, "Other": 50, "r1": 6, "r2": 5}),
                      types={"v": "nominal"})
    assert p.column("v").levels == ["a", "Other"] and p.column("v").counts == [500, 61]
    p = quiet_profile(_nominal({"a": 500, "other": 50, "r1": 6, "r2": 5}),
                      types={"v": "nominal"})
    assert p.column("v").levels == ["a", "other", "__other__"]


def test_rare_binary_level_warns_and_single_level_becomes_constant() -> None:
    df = pd.DataFrame({"b": ["y"] * 995 + ["n"] * 5, "c": ["x"] * 995 + [None] * 5})
    with pytest.warns(PrivacyWarning, match="fewer than 10"):
        p = profile_dataframe(df, types={"b": "binary"})
    assert p.column("b").counts == [5, 995]
    assert isinstance(p.column("c"), ConstantColumn) and p.column("c").value == "x"


def test_nominal_collapsing_to_one_level_is_constant() -> None:
    p = quiet_profile(_nominal({"a": 500, "r1": 3, "r2": 2}), types={"v": "nominal"})
    col = p.column("v")
    assert isinstance(col, ConstantColumn) and col.value == "a"
    assert "single_level" in warning_codes(p, "v")


# --- pseudonymization ----------------------------------------------------------------------


def test_pseudonymize_only_listed_string_columns(clinical_df) -> None:
    p = quiet_profile(clinical_df, types=RECOMMENDED_TYPES,
                      pseudonymize=["indication", "stage", "ecog"])
    ind = p.column("indication")
    assert ind.pseudonymized and ind.levels == ["L01", "L02", "L03", "L04"]
    assert p.level_map.columns["indication"]["L01"] == "surveillance"
    assert p.column("stage").levels == ["L01", "L02", "L03", "L04"]
    assert p.level_map.columns["stage"] == {"L01": "I", "L02": "II", "L03": "III", "L04": "IV"}
    assert p.column("ecog").levels == [0, 1, 2, 3, 4] and not p.column("ecog").pseudonymized
    assert p.column("sex").levels == ["F", "M"]
    text = p.to_json()
    assert "surveillance" not in text and '"III"' not in text
    assert "level_map" not in json.loads(text)


def test_level_map_roundtrip(tmp_path, clinical_df) -> None:
    p = quiet_profile(clinical_df, types=RECOMMENDED_TYPES, pseudonymize=["indication"])
    path = tmp_path / "levels.local.json"
    p.save_level_map(path)
    loaded = LevelMap.load(path)
    assert loaded.columns == p.level_map.columns
    pseudo = generate(p, seed=1)
    real = generate(p, seed=1, level_map=path)
    assert set(pseudo["indication"].dropna()) == {"L01", "L02", "L03", "L04"}
    mapped = pseudo["indication"].map(loaded.columns["indication"])
    assert (mapped.isna() == real["indication"].isna()).all()
    observed = real["indication"].notna()
    assert (mapped[observed] == real["indication"][observed]).all()


# --- non-distribution columns --------------------------------------------------------------


def test_special_columns(clinical_profile) -> None:
    assert isinstance(clinical_profile.column("patient_id"), IdColumn)
    note = clinical_profile.column("note")
    assert isinstance(note, ReasonColumn) and note.type is T.TEXT
    assert isinstance(clinical_profile.column("site"), ConstantColumn)
    assert clinical_profile.column("unused").type is T.EMPTY
    assert clinical_profile.column("exam_date").type is T.DATE
    assert clinical_profile.column("admit_dt").type is T.DATETIME
    assert clinical_profile.column("visit_time").type is T.TIME


def test_user_excluded_column() -> None:
    df = pd.DataFrame({"a": np.arange(100) + 0.5, "b": np.arange(100) * 2.0})
    p = quiet_profile(df, types={"b": "excluded"})
    assert p.column("b").type is T.EXCLUDED and "b" not in generate(p, seed=0).columns


def test_profile_is_deterministic_and_roundtrips(clinical_df, clinical_profile) -> None:
    again = quiet_profile(clinical_df, types=RECOMMENDED_TYPES)
    assert again.to_json() == clinical_profile.to_json()
    assert Profile.from_json(clinical_profile.to_json()).to_json() == clinical_profile.to_json()


def test_invalid_options() -> None:
    df = pd.DataFrame({"a": np.arange(100) + 0.5})
    with pytest.raises(ValueError):
        profile_dataframe(df, rare_threshold=0)
    with pytest.raises(ValueError):
        profile_dataframe(df.iloc[:0])
