"""Per-column marginal profiling. Each profiler also returns latent feature scores."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..schema import ColumnProfile, Scalar, WarningRecord


@dataclass
class ColumnResult:
    """Profile of one column plus what the dependence estimator needs.

    ``features`` holds one array per latent dimension (NaN where missing) with the
    interval normal scores; ``feature_cells`` the latent cell probabilities they were
    derived from (used to undo the attenuation of the correlations).
    """

    column: ColumnProfile
    features: list[np.ndarray] = field(default_factory=list)
    feature_cells: list[np.ndarray] = field(default_factory=list)
    warnings: list[WarningRecord] = field(default_factory=list)
    level_map: dict[str, Scalar] | None = None
