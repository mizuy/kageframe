"""date / datetime / time columns: profiling (k rule, resolution, formats) and generation."""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import polars as pl
import pytest

from kageframe import Profile, generate, profile_dataframe
from kageframe.marginals.temporal import date_time_parts, time_seconds
from kageframe.schema import DateColumn, DatetimeColumn, TimeColumn
from kageframe.types import ColumnType as T


def quiet_profile(df: pl.DataFrame, **kwargs) -> Profile:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return profile_dataframe(df, **kwargs)


def _days(values: pl.Series) -> np.ndarray:
    return date_time_parts(values)[0]


def _datetimes(base: dt.datetime, offsets, unit: str) -> pl.Series:
    return pl.Series([base + dt.timedelta(**{unit: int(v)}) for v in offsets])


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(2026)


# --- date ----------------------------------------------------------------------------------


def test_date_profile_uses_days_and_blurred_range(clinical_df, clinical_profile) -> None:
    col = clinical_profile.column("exam_date")
    assert isinstance(col, DateColumn) and col.output_format is None
    days = np.sort(_days(clinical_df["exam_date"].drop_nulls()))
    k = clinical_profile.options["min_tail_count"]
    assert days[k - 1] <= col.date_part.min <= days[k]
    assert days[-k - 1] <= col.date_part.max <= days[-k]
    assert col.date_part.min > days[0] or days[0] == days[k - 1]


def test_generated_dates_are_whole_days_in_range(clinical_profile) -> None:
    dummy = generate(clinical_profile, seed=1)["exam_date"].drop_nulls()
    col = clinical_profile.column("exam_date")
    assert (dummy == dummy.dt.truncate("1d")).all()
    days = _days(dummy)
    assert days.min() >= col.date_part.min and days.max() <= col.date_part.max


def test_date_strings_keep_their_format(rng) -> None:
    days = _datetimes(dt.datetime(2020, 1, 1), rng.integers(0, 700, 3000), "days")
    df = pl.DataFrame({"visit": days.dt.to_string("%Y/%m/%d")})
    p = quiet_profile(df)
    assert p.column("visit").type is T.DATE and p.column("visit").output_format == "%Y/%m/%d"
    out = generate(p, seed=0)["visit"]
    assert out.dtype == pl.String
    assert out.str.contains(r"^\d{4}/\d{2}/\d{2}$").all()


def test_date_values_are_supported(rng) -> None:
    base = dt.date(2021, 4, 1)
    df = pl.DataFrame({"d": [base + dt.timedelta(days=int(v)) for v in rng.integers(0, 300,
                                                                                     1000)]})
    p = quiet_profile(df)
    assert p.column("d").type is T.DATE and p.column("d").source_dtype == "Date"
    out = generate(p, seed=0)["d"]
    assert out.dtype == pl.Date
    assert out.min() >= base


# --- datetime ------------------------------------------------------------------------------


def test_datetime_is_split_into_date_and_time(clinical_profile) -> None:
    col = clinical_profile.column("admit_dt")
    assert isinstance(col, DatetimeColumn)
    assert col.time_part.resolution_seconds == 60  # admission times are whole minutes
    keys = [lb.key for lb in clinical_profile.latent_labels()]
    assert "admit_dt::date" in keys and "admit_dt::time" in keys


def test_generated_datetimes_follow_the_resolution(clinical_profile) -> None:
    out = generate(clinical_profile, seed=4)["admit_dt"].drop_nulls()
    assert (out.dt.second() == 0).all()
    secs = (out - out.dt.truncate("1d")).dt.total_seconds()
    col = clinical_profile.column("admit_dt")
    assert secs.min() >= col.time_part.min and secs.max() <= col.time_part.max


def test_datetime_time_of_day_pattern_is_kept(clinical_df, clinical_profile) -> None:
    real = clinical_df["admit_dt"].drop_nulls().dt.hour()
    dummy = generate(clinical_profile, seed=4)["admit_dt"].drop_nulls().dt.hour()
    assert abs(real.mean() - dummy.mean()) < 0.2
    assert abs(real.is_between(9, 17).mean() - dummy.is_between(9, 17).mean()) < 0.02


def test_datetime_with_time_zone(rng) -> None:
    ts = _datetimes(dt.datetime(2022, 3, 1, 8), rng.integers(0, 60 * 24 * 90, 2000), "minutes")
    df = pl.DataFrame({"t": ts.dt.replace_time_zone("Asia/Tokyo")})
    p = quiet_profile(df)
    assert p.column("t").tz == "Asia/Tokyo"
    out = generate(p, seed=0)["t"]
    assert out.dtype == pl.Datetime("us", "Asia/Tokyo")


def test_datetime_strings_keep_their_format(rng) -> None:
    ts = _datetimes(dt.datetime(2022, 1, 1), rng.integers(0, 10**7, 2000), "seconds")
    df = pl.DataFrame({"t": ts.dt.to_string("%Y-%m-%d %H:%M:%S")})
    p = quiet_profile(df)
    col = p.column("t")
    assert col.type is T.DATETIME and col.output_format == "%Y-%m-%d %H:%M:%S"
    assert col.time_part.resolution_seconds == 1
    out = generate(p, seed=0)["t"]
    assert out.str.contains(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all()


def test_datetime_strings_with_utc_offsets_are_converted_to_utc() -> None:
    values = ["2022-01-01T09:00:00+09:00", "2022-01-02T10:30:00Z"] * 50
    p = quiet_profile(pl.DataFrame({"t": values}))
    col = p.column("t")
    assert col.type is T.DATETIME and col.tz == "UTC"


def test_date_and_time_dependence_is_estimated(rng) -> None:
    """Later dates get later admission times; the split latent keeps the correlation."""
    n = 5000
    z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n)
    days = np.floor(300 + 100 * z[:, 0]).astype(int)
    secs = np.clip(np.round((12 * 3600 + 3 * 3600 * z[:, 1]) / 60) * 60, 0, 86340)
    base = dt.datetime(2020, 1, 1)
    ts = [base + dt.timedelta(days=int(d), seconds=int(s)) for d, s in zip(days, secs,
                                                                         strict=True)]
    p = quiet_profile(pl.DataFrame({"t": ts}))
    assert p.dependence.latent.matrix[0][1] == pytest.approx(0.6, abs=0.04)
    out = generate(p, seed=0)["t"]
    d = date_time_parts(out)
    assert np.corrcoef(d[0], d[1])[0, 1] == pytest.approx(np.corrcoef(days, secs)[0, 1],
                                                          abs=0.05)


# --- time ----------------------------------------------------------------------------------


def test_time_strings(clinical_df, clinical_profile) -> None:
    col = clinical_profile.column("visit_time")
    assert isinstance(col, TimeColumn) and col.output_format == "%H:%M"
    assert col.time_part.resolution_seconds == 60
    out = generate(clinical_profile, seed=0)["visit_time"].drop_nulls()
    assert out.str.contains(r"^\d{2}:\d{2}$").all()
    real = time_seconds(clinical_df["visit_time"].drop_nulls())
    assert abs(time_seconds(out).mean() - real.mean()) < 0.05 * real.std()


def test_time_values_and_durations(rng) -> None:
    secs = rng.integers(8 * 3600, 18 * 3600, 2000) // 60 * 60
    times = [dt.time(int(s // 3600), int(s % 3600 // 60)) for s in secs]
    durations = [dt.timedelta(seconds=int(s)) for s in secs]
    p = quiet_profile(pl.DataFrame({"t": times, "d": durations}))
    assert p.column("t").type is T.TIME and p.column("d").type is T.TIME
    out = generate(p, seed=0)
    assert out["t"].dtype == pl.Time
    assert out["d"].dtype.base_type() is pl.Duration
    assert out["d"].dt.total_seconds().is_between(8 * 3600, 18 * 3600).all()


def test_hourly_resolution_is_detected(rng) -> None:
    hours = rng.integers(0, 24, 1000)
    p = quiet_profile(pl.DataFrame({"t": [f"{h:02d}:00" for h in hours]}))
    assert p.column("t").time_part.resolution_seconds == 3600
    assert generate(p, seed=0)["t"].str.ends_with(":00").all()


# --- edge cases ----------------------------------------------------------------------------


def test_too_few_dates_are_generated_as_missing() -> None:
    values = [dt.date(2020, 1, 1), dt.date(2020, 2, 1), dt.date(2020, 3, 1)] * 5 + [None] * 85
    p = quiet_profile(pl.DataFrame({"d": values}))
    assert p.column("d").date_part.grid is None
    assert any(w.code == "too_few_values" for w in p.warnings)
    assert generate(p, seed=0)["d"].is_null().all()


def test_constant_date_column() -> None:
    p = quiet_profile(pl.DataFrame({"d": [dt.datetime(2020, 5, 5)] * 50}))
    assert p.column("d").type is T.CONSTANT
    out = generate(p, seed=0)["d"]
    assert out.dtype == pl.Datetime("us")
    assert (out == dt.datetime(2020, 5, 5)).all()


def test_temporal_profile_roundtrips_through_json(clinical_profile) -> None:
    again = Profile.from_json(clinical_profile.to_json())
    for name in ("exam_date", "admit_dt", "visit_time"):
        assert again.column(name) == clinical_profile.column(name)
