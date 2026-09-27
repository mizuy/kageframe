import copy
import json
import warnings

import pytest

from conftest import build_example_columns, build_example_profile
from kageframe import SCHEMA_VERSION, LevelMap, Profile, ProfileSchemaError, load_profile
from kageframe.exceptions import KageFrameWarning
from kageframe.schema import (
    CategoricalColumn,
    LatentLabel,
    NumericColumn,
    Quantiles,
    latent_labels,
)
from kageframe.types import ColumnType as T


def test_roundtrip_dict_json_and_file(example_profile, tmp_path):
    d = example_profile.to_dict()
    assert Profile.from_dict(d).to_dict() == d
    assert Profile.from_json(example_profile.to_json()) == example_profile
    path = tmp_path / "profile.json"
    example_profile.save(path)
    loaded = load_profile(path)
    assert loaded == example_profile
    assert loaded.to_json() == example_profile.to_json()


def test_every_column_type_roundtrips(example_profile):
    types = {c.type for c in example_profile.columns}
    assert types == set(T)
    for col in example_profile.columns:
        restored = Profile.from_dict(example_profile.to_dict()).column(col.name)
        assert restored == col
        assert type(restored) is type(col)


def test_column_order_is_preserved(example_profile):
    names = [c.name for c in build_example_columns()]
    assert Profile.from_json(example_profile.to_json()).column_names == names


def test_json_is_deterministic_and_strict():
    a = build_example_profile().to_json()
    b = build_example_profile().to_json()
    assert a == b
    assert "created_at" not in a
    assert json.loads(a)["schema_version"] == SCHEMA_VERSION


def test_output_columns_exclude_text_and_excluded(example_profile):
    out = example_profile.output_columns
    assert "note" not in out and "dropped" not in out
    assert "patient_id" in out and "unused" in out


def test_float_values_roundtrip_exactly(example_profile):
    restored = Profile.from_json(example_profile.to_json())
    col = restored.column("indication")
    assert col.probabilities == example_profile.column("indication").probabilities


# -- level map --------------------------------------------------------------------


def test_level_map_is_never_in_profile_json(example_profile, tmp_path):
    text = example_profile.to_json()
    for real in ("surveillance", "screening", '"M"', '"F"'):
        assert real not in text
    assert "level_map" not in json.loads(text)
    assert Profile.from_json(text).level_map is None


def test_level_map_save_and_load(example_profile, tmp_path):
    path = tmp_path / "level_map.local.json"
    example_profile.save_level_map(path)
    lm = LevelMap.load(path)
    assert lm == example_profile.level_map
    assert "DO NOT SHARE" in path.read_text()


def test_level_map_and_profile_are_not_interchangeable(example_profile, tmp_path):
    lm_path, prof_path = tmp_path / "lm.json", tmp_path / "p.json"
    example_profile.save_level_map(lm_path)
    example_profile.save(prof_path)
    with pytest.raises(ProfileSchemaError, match="level map"):
        load_profile(lm_path)
    with pytest.raises(ProfileSchemaError, match="not a kageframe level map"):
        LevelMap.load(prof_path)


def test_save_level_map_without_map_raises():
    with pytest.raises(ValueError):
        build_example_profile(with_level_map=False).save_level_map("x.json")


def test_level_map_must_match_pseudonymized_columns():
    p = build_example_profile(with_level_map=False)
    with pytest.raises(ProfileSchemaError, match="not a pseudonymized"):
        Profile(p.n_rows, p.columns, level_map=LevelMap({"ecog": {"L01": 0}}))
    with pytest.raises(ProfileSchemaError, match="not among"):
        Profile(p.n_rows, p.columns, level_map=LevelMap({"sex": {"L09": "M"}}))
    with pytest.raises(ProfileSchemaError, match="unique"):
        LevelMap.from_dict({**LevelMap({"sex": {"L01": "M", "L02": "M"}}).to_dict()})


# -- versions and strictness ------------------------------------------------------


def test_major_version_mismatch_raises(example_profile):
    d = example_profile.to_dict()
    d["schema_version"] = "1.0.0"
    with pytest.raises(ProfileSchemaError, match="incompatible"):
        Profile.from_dict(d)


def test_newer_minor_version_warns(example_profile):
    d = example_profile.to_dict()
    d["schema_version"] = "0.9.0"
    with pytest.warns(KageFrameWarning, match="newer"):
        Profile.from_dict(d)


def test_unknown_and_missing_fields(example_profile):
    d = example_profile.to_dict()
    d["columns"][1]["secret"] = 1
    with pytest.raises(ProfileSchemaError, match=r"columns\[1\]: unknown fields"):
        Profile.from_dict(d)
    d = example_profile.to_dict()
    del d["columns"][3]["probabilities"]
    with pytest.raises(ProfileSchemaError, match="missing required"):
        Profile.from_dict(d)
    d = example_profile.to_dict()
    d["options"]["typo"] = 1
    with pytest.raises(ProfileSchemaError, match="unknown options"):
        Profile.from_dict(d)


def test_non_finite_numbers_rejected(example_profile):
    with pytest.raises(ProfileSchemaError, match="non-finite"):
        Profile.from_json(example_profile.to_json().replace('"mean": 65.2', '"mean": NaN'))
    col = copy.deepcopy(example_profile.column("age"))
    col.mean = float("nan")
    with pytest.raises(ProfileSchemaError):
        Profile(10, [col])


def test_invalid_json_text():
    with pytest.raises(ProfileSchemaError, match="invalid JSON"):
        Profile.from_json("{not json")


@pytest.mark.parametrize(
    "column, key, value, match",
    [
        (3, "levels", ["L01"], "exactly 2"),
        (3, "probabilities", [0.5, 0.6], "sum to 1"),
        (3, "thresholds", [0.1, 0.2], "expected 1 thresholds"),
        (4, "thresholds", [1.0, 0.5, 2.0], "non-decreasing"),
        (5, "other_level", "Missing", "one of the levels"),
        (5, "thresholds", [0.0, 1.0], "no thresholds"),
        (1, "grid", {"probs": [0, 0.5, 1], "values": [3, 2, 1]}, "non-decreasing"),
        (1, "grid", {"probs": [0, 0, 1], "values": [1, 2, 3]}, "strictly increasing"),
        (1, "support", [1, 2], "must not have support"),
        (1, "min", 100.0, "min must be <= max"),
        (2, "support", [0.0, 0.0, 2.0, 3.0], "unique"),
        (1, "missing_rate", 1.5, "outside"),
        (0, "value_kind", "uuid", "expected one of"),
        (6, "reference", "2000-01-01", "expected one of"),
        (1, "type", "float", "unknown column type"),
    ],
)
def test_invalid_column_fields(example_profile, column, key, value, match):
    d = example_profile.to_dict()
    d["columns"][column][key] = value
    with pytest.raises(ProfileSchemaError, match=match):
        Profile.from_dict(d)


def test_column_class_must_match_type():
    col = build_example_columns()[1]
    col.type = T.BINARY
    with pytest.raises(ProfileSchemaError, match="cannot have type"):
        Profile(1000, [col])


def test_profile_level_checks():
    cols = build_example_columns()
    with pytest.raises(ProfileSchemaError, match="duplicate"):
        Profile(1000, [cols[1], cols[1]])
    with pytest.raises(ProfileSchemaError, match="exceeds n_rows"):
        Profile(10, [cols[1]])


# -- latent labels and dependence -------------------------------------------------


def test_latent_label_order(example_profile):
    keys = [lb.key for lb in example_profile.latent_labels()]
    assert keys == [
        "age", "pain_score", "sex", "ecog",
        "indication[L01]", "indication[L02]", "indication[Other]",
        "exam_date", "admit_dt::date", "admit_dt::time", "visit_time",
    ]


def test_label_keys_and_parsing():
    lb = LatentLabel("a[b]", "level", 3)
    assert lb.key == "a[b][3]"
    assert LatentLabel.from_dict(lb.to_dict(), "x") == lb
    bad = {**lb.to_dict(), "key": "a[b][4]"}
    with pytest.raises(ProfileSchemaError, match="key"):
        LatentLabel.from_dict(bad, "x")


def test_latent_labels_skip_columns_without_latent():
    cols = build_example_columns()
    names = {lb.column for lb in latent_labels(cols)}
    assert names.isdisjoint({"patient_id", "note", "site", "unused", "dropped"})


def test_dependence_label_mismatch(example_profile):
    d = example_profile.to_dict()
    d["dependence"]["latent"]["labels"].reverse()
    with pytest.raises(ProfileSchemaError, match="labels do not match"):
        Profile.from_dict(d)


def test_latent_nominal_block_must_be_identity(example_profile):
    d = example_profile.to_dict()
    m = d["dependence"]["latent"]["matrix"]
    m[4][5] = m[5][4] = 0.1
    with pytest.raises(ProfileSchemaError, match="uncorrelated"):
        Profile.from_dict(d)


def test_matrix_structure_checks(example_profile):
    d = example_profile.to_dict()
    d["dependence"]["latent"]["matrix"][0][1] = 0.9
    with pytest.raises(ProfileSchemaError, match="symmetric"):
        Profile.from_dict(d)
    d = example_profile.to_dict()
    d["dependence"]["latent"]["matrix"][0][0] = 0.9
    with pytest.raises(ProfileSchemaError, match="diagonal"):
        Profile.from_dict(d)
    d = example_profile.to_dict()
    d["dependence"]["observed"]["matrix"][0][1] = None
    d["dependence"]["observed"]["matrix"][1][0] = None
    with pytest.raises(ProfileSchemaError, match="null is only allowed"):
        Profile.from_dict(d)
    d = example_profile.to_dict()
    d["dependence"]["latent"]["matrix"][0][1] = 1.5
    d["dependence"]["latent"]["matrix"][1][0] = 1.5
    with pytest.raises(ProfileSchemaError, match="outside"):
        Profile.from_dict(d)
    d = example_profile.to_dict()
    del d["dependence"]["observed"]["n_pairs"]
    with pytest.raises(ProfileSchemaError, match="n_pairs"):
        Profile.from_dict(d)


def test_profile_without_dependence_roundtrips():
    p = build_example_profile(with_level_map=False)
    p.dependence = None
    assert Profile.from_json(p.to_json()) == p


def test_quantiles_need_two_points():
    with pytest.raises(ProfileSchemaError):
        Quantiles.from_dict({"probs": [0.5], "values": [1.0]}, "q")


def test_direct_construction_is_validated():
    with pytest.raises(ProfileSchemaError):
        Profile(100, [CategoricalColumn("x", T.NOMINAL, "str", 10, 0.0, levels=["a"],
                                        counts=[10], probabilities=[1.0])])
    with pytest.raises(ProfileSchemaError):
        Profile(100, [NumericColumn("x", T.NUMERIC, "float64", 10, 0.0, mode="discrete",
                                    support=[1.0], probabilities=[0.5])])


def test_no_warnings_on_normal_load(example_profile):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Profile.from_json(example_profile.to_json())
