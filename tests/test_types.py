import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest

import clinmock
from clinmock.datasets import EXPECTED_EDGE_CASE_TYPES, EXPECTED_INFERRED_TYPES
from clinmock.exceptions import PrivacyWarning, TypeInferenceWarning, TypeSpecError
from clinmock.types import ColumnType as T
from clinmock.types import infer_column_type, resolve_types


def infer(values, name="x", dtype=None) -> T:
    spec, _ = infer_column_type(pd.Series(values, name=name, dtype=dtype))
    return spec.type


def infer_warn_codes(values, name="x", dtype=None) -> list[str]:
    _, warns = infer_column_type(pd.Series(values, name=name, dtype=dtype))
    return [w.code for w in warns]


# -- inference rules (plan §2.2) ------------------------------------------------


def test_fixture_types_are_inferred(clinical_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inferred = clinmock.infer_types(clinical_df)
    assert inferred == {**EXPECTED_INFERRED_TYPES, **EXPECTED_EDGE_CASE_TYPES}


def test_empty_and_constant():
    assert infer([np.nan, np.nan]) is T.EMPTY
    assert infer([None, None], dtype=object) is T.EMPTY
    assert infer(["a", "a", None]) is T.CONSTANT
    assert infer([5, 5, 5]) is T.CONSTANT


def test_bool_is_binary():
    assert infer([True, False, True]) is T.BINARY
    assert infer([True, False, None], dtype="boolean") is T.BINARY
    assert infer([True, False, None], dtype=object) is T.BINARY


def test_integer_zero_one_is_binary():
    assert infer([0, 1, 1, 0]) is T.BINARY
    assert infer([0.0, 1.0, np.nan]) is T.BINARY
    assert infer([0, 1, None], dtype="Int64") is T.BINARY


def test_integer_codes_up_to_ten_levels_are_ordinal_with_warning():
    spec, warns = infer_column_type(pd.Series([1, 2, 3, 2, 1] * 20, name="grade"))
    assert spec.type is T.ORDINAL
    assert spec.levels == (1, 2, 3)
    assert [w.code for w in warns] == ["integer_code_as_ordinal"]
    assert infer([1, 2] * 10) is T.ORDINAL
    assert infer(list(range(10)) * 5) is T.ORDINAL
    assert infer([1.0, 2.0, 3.0, np.nan]) is T.ORDINAL


def test_integer_with_more_than_ten_levels_is_numeric():
    assert infer(list(range(11)) * 5) is T.NUMERIC
    assert infer([0.5, 1.0, 1.5]) is T.NUMERIC
    assert infer(np.random.default_rng(0).normal(size=200)) is T.NUMERIC


def test_integer_identifier_is_id():
    assert infer(np.arange(1000, 1200)) is T.ID
    assert infer_warn_codes(np.arange(1000, 1200)) == ["id_like_excluded"]
    shuffled = np.random.default_rng(0).permutation(np.arange(200))
    assert infer(shuffled) is T.ID
    assert infer(np.random.default_rng(0).choice(10**9, 200, replace=False)) is T.NUMERIC


def test_categorical_dtypes():
    ordered = pd.Categorical(["lo", "hi", "mid"], categories=["lo", "mid", "hi"], ordered=True)
    spec, _ = infer_column_type(pd.Series(ordered))
    assert spec.type is T.ORDINAL
    assert spec.levels == ("lo", "mid", "hi")
    assert infer(pd.Categorical(["a", "b", "a"])) is T.BINARY
    assert infer(pd.Categorical(["a", "b", "c"])) is T.NOMINAL


def test_datetime_dtypes():
    assert infer(pd.to_datetime(["2020-01-01", "2020-02-03", None])) is T.DATE
    assert infer(pd.to_datetime(["2020-01-01 10:00", "2020-02-03 00:00"])) is T.DATETIME
    tz = pd.to_datetime(["2020-01-01 10:00", "2020-01-02 11:00"]).tz_localize("Asia/Tokyo")
    assert infer(tz) is T.DATETIME
    assert infer(pd.to_timedelta(["1h", "2h"])) is T.TIME


def test_python_temporal_objects():
    assert infer([dt.date(2020, 1, 1), dt.date(2021, 5, 2)], dtype=object) is T.DATE
    assert infer([dt.datetime(2020, 1, 1, 9), dt.datetime(2021, 5, 2)], dtype=object) \
        is T.DATETIME
    assert infer([dt.time(9, 30), dt.time(14, 0)], dtype=object) is T.TIME


def test_temporal_strings():
    assert infer(["2020-01-01", "2020/02/03", None]) is T.DATE
    assert infer(["2020-01-01 10:00", "2020-02-03 23:59:59"]) is T.DATETIME
    assert infer(["2020-01-01T00:00:00", "2020-02-03T00:00:00"]) is T.DATE
    assert infer(["09:30", "14:05", "23:59"]) is T.TIME
    assert infer(["09:30:10", "14:05:00"]) is T.TIME
    assert infer(["25:61", "14:05"] * 5) is not T.TIME
    assert infer(["2020-01-01"] * 3 + ["unknown"] * 7 + ["n/a"]) is T.NOMINAL


def test_string_categories():
    assert infer(["M", "F", "F", None]) is T.BINARY
    assert infer(["a", "b", "c"] * 5) is T.NOMINAL
    assert infer([f"lvl{i}" for i in range(50)] * 2) is T.NOMINAL


def test_string_identifier_and_text():
    ids = [f"P{i:07d}" for i in range(500)]
    assert infer(ids) is T.ID
    text = [f"patient reports pain for {i} days, follow up" for i in range(500)]
    assert infer(text) is T.TEXT
    assert infer_warn_codes(text) == ["free_text_excluded"]
    long_few = [f"very long free text sentence number {i} " * 3 for i in range(40)]
    assert infer(long_few) is T.TEXT
    high_card_short = [f"code {i % 300}" for i in range(600)]
    assert infer(high_card_short) is T.TEXT


def test_inference_itself_never_warns_about_pseudonymization():
    assert infer_warn_codes(["Dr. A", "Dr. B", "Dr. C"] * 5, name="doctor") == []


def test_mixed_object_values_do_not_crash():
    assert infer(["a", 1, 2.5, "a", 1], dtype=object) is T.NOMINAL


# -- overrides -------------------------------------------------------------------


@pytest.fixture
def df():
    return pd.DataFrame({
        "stage": ["I", "II", "III", "IV", None] * 4,
        "code": [1, 2, 3, 2, 1] * 4,
        "sex": ["M", "F", "F", "M", "F"] * 4,
        "val": np.linspace(0, 1, 20),
    })


def test_override_string_and_dict_forms(df):
    res = resolve_types(df, {"code": "nominal", "stage": {"type": "ordinal",
                                                        "levels": ["I", "II", "III", "IV"]},
                             "val": clinmock.ColumnType.NUMERIC})
    assert res.specs["code"].type is T.NOMINAL
    assert res.specs["code"].source == "override"
    assert res.specs["stage"].levels == ("I", "II", "III", "IV")
    assert res.specs["sex"].source == "inferred"
    assert all(w.column != "code" for w in res.warnings)


def test_override_takes_precedence_over_inference(df):
    assert resolve_types(df).specs["code"].type is T.ORDINAL
    assert resolve_types(df, {"code": "numeric"}).specs["code"].type is T.NUMERIC


def test_override_ordinal_without_levels(df):
    assert resolve_types(df, {"code": "ordinal"}).specs["code"].levels == (1, 2, 3)
    with pytest.raises(TypeSpecError, match="order is never guessed"):
        resolve_types(df, {"stage": "ordinal"})


def test_override_levels_must_cover_data(df):
    with pytest.raises(TypeSpecError, match="not listed"):
        resolve_types(df, {"stage": {"type": "ordinal", "levels": ["I", "II", "III"]}})
    res = resolve_types(df, {"code": {"type": "ordinal", "levels": [1.0, 2.0, 3.0, 4.0]}})
    assert res.specs["code"].levels == (1.0, 2.0, 3.0, 4.0)


@pytest.mark.parametrize(
    "override, match",
    [
        ("categorical", "unknown type"),
        ("constant", "derived from the data"),
        ({"levels": ["a"]}, "needs a 'type'"),
        ({"type": "nominal", "order": []}, "unknown override keys"),
        ({"type": "numeric", "levels": [1, 2]}, "only valid"),
        ({"type": "nominal", "levels": "abc"}, "must be a list"),
        ({"type": "nominal", "levels": ["a", "a"]}, "duplicates"),
        ({"type": "binary", "levels": ["M", "F", "X"]}, "exactly 2"),
        ({"type": "nominal", "pseudonymize": "yes"}, "must be a bool"),
        ({"type": "nominal", "keep_level_names": True}, "unknown override keys"),
    ],
)
def test_invalid_overrides(df, override, match):
    with pytest.raises(TypeSpecError, match=match):
        resolve_types(df, {"sex": override})


def test_binary_override_rejects_three_values(df):
    with pytest.raises(TypeSpecError, match="3 distinct"):
        resolve_types(df, {"code": "binary"})


def test_numeric_override_rejects_strings(df):
    with pytest.raises(TypeSpecError, match="not numeric"):
        resolve_types(df, {"sex": "numeric"})
    ok = pd.DataFrame({"x": ["1", "2.5", None]})
    assert resolve_types(ok, {"x": "numeric"}).specs["x"].type is T.NUMERIC


def test_unknown_columns_rejected(df):
    with pytest.raises(TypeSpecError, match="unknown columns"):
        resolve_types(df, {"nope": "numeric"})
    with pytest.raises(TypeSpecError, match="unknown columns"):
        resolve_types(df, pseudonymize=["nope"])


def test_column_name_checks():
    with pytest.raises(TypeSpecError, match="duplicate"):
        resolve_types(pd.DataFrame([[1, 2]], columns=["a", "a"]))
    with pytest.raises(TypeSpecError, match="strings"):
        resolve_types(pd.DataFrame([[1, 2]]))


# -- pseudonymization (opt-in) ----------------------------------------------------


def codes_for(res, column):
    return [w.code for w in res.warnings if w.column == column]


def test_level_names_are_kept_by_default(df):
    res = resolve_types(df)
    assert not any(s.pseudonymize for s in res.specs.values())


def test_pseudonymize_list_and_override(df):
    res = resolve_types(df, {"stage": {"type": "ordinal", "levels": ["I", "II", "III", "IV"],
                                       "pseudonymize": True}},
                        pseudonymize=["sex"])
    assert res.specs["sex"].pseudonymize is True
    assert res.specs["stage"].pseudonymize is True
    assert res.specs["code"].pseudonymize is False
    res = resolve_types(df, {"sex": {"type": "binary", "pseudonymize": False}},
                        pseudonymize=["sex"])
    assert res.specs["sex"].pseudonymize is False


def test_pseudonymize_argument_validation(df):
    with pytest.raises(TypeSpecError, match="list of column names"):
        resolve_types(df, pseudonymize="sex")
    with pytest.raises(TypeSpecError, match="only valid"):
        resolve_types(df, {"val": {"type": "numeric", "pseudonymize": True}})


def test_codes_and_booleans_are_never_pseudonymized():
    frame = pd.DataFrame({"code": [1, 2, 3] * 10, "flag01": [0, 1] * 15,
                          "flag": [True, False] * 15,
                          "cat": pd.Categorical([1, 2, 3] * 10)})
    res = resolve_types(frame, {"cat": "nominal"}, pseudonymize=list(frame.columns))
    for name in frame.columns:
        assert res.specs[name].pseudonymize is False
        assert codes_for(res, name)[-1] == "pseudonymize_ignored"


def test_pseudonymize_on_non_categorical_inferred_column_is_ignored_with_warning(df):
    res = resolve_types(df, pseudonymize=["val"])
    assert res.specs["val"].pseudonymize is False
    assert codes_for(res, "val") == ["pseudonymize_ignored"]


@pytest.mark.parametrize("name", ["hospital", "doctor", "facility_code", "site", "center",
                                  "institution", "hospitalName", "attending_dr", "siteID",
                                  "patient_name", "病院名", "担当医師", "ward"])
def test_sensitive_names_warn_to_consider_pseudonymization(name):
    frame = pd.DataFrame({name: ["a", "b", "c"] * 10})
    res = resolve_types(frame)
    assert codes_for(res, name) == ["consider_pseudonymize"]
    assert res.specs[name].pseudonymize is False
    assert all(w.category is PrivacyWarning for w in res.warnings)
    assert codes_for(resolve_types(frame, pseudonymize=[name]), name) == []


@pytest.mark.parametrize("name", ["indication", "stage", "sex", "idea", "consideration",
                                  "valid_flag", "insider"])
def test_ordinary_names_do_not_warn(name):
    frame = pd.DataFrame({name: ["a", "b", "c"] * 10})
    assert codes_for(resolve_types(frame), name) == []


def test_high_cardinality_string_categorical_warns():
    frame = pd.DataFrame({"x": [f"v{i}" for i in range(30)]})
    res = resolve_types(frame)
    assert res.specs["x"].type is T.NOMINAL
    assert codes_for(res, "x") == ["consider_pseudonymize"]
    assert "distinct value" in res.warnings[0].message


def test_sensitive_name_on_integer_codes_does_not_warn():
    frame = pd.DataFrame({"hospital": [1, 2, 3] * 10})
    assert "consider_pseudonymize" not in codes_for(resolve_types(frame), "hospital")


def test_fixture_warnings(clinical_df):
    res = resolve_types(clinical_df)
    assert not any(w.code == "consider_pseudonymize" for w in res.warnings)
    assert not any(s.pseudonymize for s in res.specs.values())


def test_infer_types_emits_python_warnings(df):
    with pytest.warns(TypeInferenceWarning, match="code"):
        out = clinmock.infer_types(df)
    assert out == {"stage": "nominal", "code": "ordinal", "sex": "binary", "val": "numeric"}


def test_privacy_warning_category():
    frame = pd.DataFrame({"pid": [f"P{i:06d}" for i in range(300)]})
    with pytest.warns(PrivacyWarning):
        clinmock.infer_types(frame)
