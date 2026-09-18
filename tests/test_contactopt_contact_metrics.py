import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from contactopt_contact_metrics import sanitized_contact_mean


def test_sanitized_contact_mean_treats_undefined_values_as_zero():
    mean, finite_fraction = sanitized_contact_mean(
        np.asarray([1.0, np.nan, 0.0], dtype=np.float64)
    )

    assert mean == 1.0 / 3.0
    assert finite_fraction == 2.0 / 3.0


def test_sanitized_contact_mean_handles_all_invalid_values():
    mean, finite_fraction = sanitized_contact_mean(
        np.asarray([np.nan, np.inf, -np.inf], dtype=np.float64)
    )

    assert mean == 0.0
    assert finite_fraction == 0.0
