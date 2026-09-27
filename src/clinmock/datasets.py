"""Synthetic "clinical-like" datasets with a known latent Gaussian copula.

The data are generated entirely from random numbers; they contain no real patient
information. They are used as test fixtures and in examples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

LATENT_LABELS: tuple[str, ...] = (
    "age",
    "bmi",
    "crp",
    "sex",
    "smoker",
    "stage",
    "ecog",
    "indication[screening]",
    "indication[surveillance]",
    "indication[FIT_positive]",
    "indication[diagnostic]",
    "exam_date",
    "admit_dt::date",
    "admit_dt::time",
    "visit_time",
    "lab_flag",
)

# Factor loadings; the four indication scores load on distinct factors (or none) so
# that the within-nominal block of the latent correlation matrix is the identity.
_LOADINGS: dict[str, tuple[float, float, float, float]] = {
    "age": (0.6, 0.0, 0.0, 0.0),
    "bmi": (-0.2, 0.4, 0.0, 0.0),
    "crp": (0.4, 0.0, 0.0, 0.3),
    "sex": (0.0, 0.5, 0.0, 0.0),
    "smoker": (0.3, 0.5, 0.0, 0.0),
    "stage": (0.5, 0.0, 0.0, 0.0),
    "ecog": (0.6, 0.0, 0.0, 0.0),
    "indication[screening]": (0.0, 0.0, 0.0, 0.0),
    "indication[surveillance]": (0.4, 0.0, 0.0, 0.0),
    "indication[FIT_positive]": (0.0, 0.4, 0.0, 0.0),
    "indication[diagnostic]": (0.0, 0.0, 0.0, 0.5),
    "exam_date": (0.0, 0.0, 0.6, 0.0),
    "admit_dt::date": (0.0, 0.0, 0.6, 0.0),
    "admit_dt::time": (0.0, 0.0, 0.4, 0.0),
    "visit_time": (0.0, 0.0, 0.3, 0.4),
    "lab_flag": (0.0, 0.4, 0.0, 0.0),
}

MISSING_RATES: dict[str, float] = {
    "patient_id": 0.0,
    "age": 0.01,
    "bmi": 0.05,
    "crp": 0.20,
    "sex": 0.0,
    "smoker": 0.03,
    "stage": 0.10,
    "ecog": 0.08,
    "indication": 0.02,
    "exam_date": 0.05,
    "admit_dt": 0.10,
    "visit_time": 0.15,
    "lab_flag": 0.12,
    "note": 0.30,
}

INDICATION_LEVELS = ("screening", "surveillance", "FIT_positive", "diagnostic")
INDICATION_PROBS = (0.20, 0.42, 0.15, 0.23)
RARE_INDICATIONS = {"research": 4, "other_referral": 3}
STAGE_LEVELS = ("I", "II", "III", "IV")

EXPECTED_INFERRED_TYPES: dict[str, str] = {
    "patient_id": "id",
    "age": "numeric",
    "bmi": "numeric",
    "crp": "numeric",
    "sex": "binary",
    "smoker": "binary",
    "stage": "nominal",
    "ecog": "ordinal",
    "indication": "nominal",
    "exam_date": "date",
    "admit_dt": "datetime",
    "visit_time": "time",
    "lab_flag": "binary",
    "note": "text",
}
EXPECTED_EDGE_CASE_TYPES: dict[str, str] = {"site": "constant", "unused": "empty"}

RECOMMENDED_TYPES: dict[str, object] = {
    "stage": {"type": "ordinal", "levels": list(STAGE_LEVELS)},
}

_EPOCH = np.datetime64("2019-01-01", "D")
_N_DAYS = 1461


def latent_correlation() -> tuple[list[str], np.ndarray]:
    """Return the true latent correlation matrix used by :func:`make_clinical_like_df`."""
    labels = list(LATENT_LABELS)
    loadings = np.array([_LOADINGS[label] for label in labels])
    corr = loadings @ loadings.T
    np.fill_diagonal(corr, 1.0)
    return labels, corr


def _cut(z: np.ndarray, cumulative: tuple[float, ...]) -> np.ndarray:
    return np.searchsorted(stats.norm.ppf(cumulative), z, side="right")


def make_clinical_like_df(
    n: int = 10000,
    seed: int = 0,
    *,
    missing: bool = True,
    edge_cases: bool = False,
) -> pd.DataFrame:
    """Generate a DataFrame containing every column type supported by clinmock.

    Columns: ``patient_id`` (id), ``age``/``bmi``/``crp`` (numeric; integer ties,
    one decimal, point mass at zero), ``sex`` (binary strings), ``smoker`` (binary
    0/1), ``stage`` (ordinal strings; needs a ``types=`` override), ``ecog``
    (integer-coded ordinal), ``indication`` (nominal with two rare levels),
    ``exam_date`` (date), ``admit_dt`` (datetime), ``visit_time`` (``HH:MM``
    strings), ``lab_flag`` (bool) and ``note`` (free text). With
    ``edge_cases=True`` a constant column ``site`` and an all-missing column
    ``unused`` are appended.
    """
    if n < 100:
        raise ValueError("n must be at least 100")
    rng = np.random.default_rng(seed)
    labels, corr = latent_correlation()
    z = rng.multivariate_normal(np.zeros(len(labels)), corr, size=n, method="cholesky")
    col = {label: z[:, i] for i, label in enumerate(labels)}
    u = {label: stats.norm.cdf(values) for label, values in col.items()}

    patient_id = np.array([f"P{v:07d}" for v in rng.choice(10**7, size=n, replace=False)])

    age = np.clip(np.round(65 + 12 * col["age"]), 18, 99).astype(np.int64)
    bmi = np.round(np.exp(3.17 + 0.15 * col["bmi"]), 1)
    u_crp = u["crp"]
    positive = u_crp >= 0.3
    crp = np.zeros(n)
    crp[positive] = np.round(
        np.exp(-0.5 + 1.2 * stats.norm.ppf((u_crp[positive] - 0.3) / 0.7)), 2
    )

    sex = np.where(col["sex"] > stats.norm.ppf(0.48), "M", "F").astype(object)
    smoker = (col["smoker"] > stats.norm.ppf(0.8)).astype(np.int64)
    stage = np.array(STAGE_LEVELS, dtype=object)[_cut(col["stage"], (0.25, 0.60, 0.85))]
    ecog = _cut(col["ecog"], (0.40, 0.70, 0.88, 0.97)).astype(np.int64)

    scores = np.column_stack([u[f"indication[{lv}]"] for lv in INDICATION_LEVELS])
    gumbel = -np.log(-np.log(np.clip(scores, 1e-12, 1 - 1e-12)))
    chosen = np.argmax(np.log(INDICATION_PROBS) + gumbel, axis=1)
    indication = np.array(INDICATION_LEVELS, dtype=object)[chosen]
    rare_rows = rng.choice(n, size=sum(RARE_INDICATIONS.values()), replace=False)
    start = 0
    for level, count in RARE_INDICATIONS.items():
        indication[rare_rows[start : start + count]] = level
        start += count

    exam_days = np.floor(u["exam_date"] * _N_DAYS).astype(np.int64)
    exam_date = pd.to_datetime(_EPOCH + exam_days)
    admit_days = np.floor(u["admit_dt::date"] * _N_DAYS).astype(np.int64)
    admit_seconds = np.clip(
        np.round(stats.norm.ppf(u["admit_dt::time"], 13 * 3600, 3.5 * 3600) / 60) * 60,
        0,
        86340,
    ).astype(np.int64)
    admit_dt = pd.to_datetime(_EPOCH + admit_days) + pd.to_timedelta(admit_seconds, unit="s")
    visit_minutes = np.clip(
        np.round(stats.norm.ppf(u["visit_time"], 10.5 * 60, 90)), 6 * 60, 20 * 60
    ).astype(np.int64)
    visit_time = np.array([f"{m // 60:02d}:{m % 60:02d}" for m in visit_minutes], dtype=object)
    lab_flag = col["lab_flag"] > stats.norm.ppf(0.9)

    symptoms = np.array(["abdominal pain", "bloody stool", "weight loss", "constipation",
                         "anemia on routine labs", "change in bowel habits"], dtype=object)
    plans = np.array(["colonoscopy scheduled", "CT ordered", "watchful waiting",
                      "referred to surgery", "repeat labs"], dtype=object)
    days = rng.integers(1, 120, size=n)
    weeks = rng.integers(1, 13, size=n)
    sym = symptoms[rng.integers(0, len(symptoms), size=n)]
    pln = plans[rng.integers(0, len(plans), size=n)]
    note = np.array(
        [f"Patient reports {s} for {d} days; {p}. Follow-up in {w} weeks."
         for s, d, p, w in zip(sym, days, pln, weeks, strict=True)],
        dtype=object,
    )

    df = pd.DataFrame(
        {
            "patient_id": patient_id,
            "age": age,
            "bmi": bmi,
            "crp": crp,
            "sex": sex,
            "smoker": smoker,
            "stage": stage,
            "ecog": ecog,
            "indication": indication,
            "exam_date": exam_date,
            "admit_dt": admit_dt,
            "visit_time": visit_time,
            "lab_flag": lab_flag,
            "note": note,
        }
    )

    if missing:
        for name, rate in MISSING_RATES.items():
            k = int(round(rate * n))
            if k == 0:
                continue
            mask = np.zeros(n, dtype=bool)
            mask[rng.choice(n, size=k, replace=False)] = True
            series = df[name]
            if pd.api.types.is_bool_dtype(series):
                series = series.astype("boolean")
            elif pd.api.types.is_integer_dtype(series):
                series = series.astype("Int64")
            df[name] = series.mask(mask)

    if edge_cases:
        df["site"] = "A"
        df["unused"] = np.full(n, np.nan)

    return df
