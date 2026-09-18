"""Metric helpers for ContactOpt capsule contact weights."""

import numpy as np


def sanitized_contact_mean(contact):
    """Treat undefined capsule contacts as zero and report finite support."""
    values = np.asarray(contact, dtype=np.float64).reshape(-1)
    finite_fraction = float(np.isfinite(values).mean())
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return float(values.mean()), finite_fraction
