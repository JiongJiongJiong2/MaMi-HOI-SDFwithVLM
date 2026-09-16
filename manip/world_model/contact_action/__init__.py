"""Action-conditioned short-horizon contact/response dynamics."""

from .features import (
    ACTION_DIM,
    CONTACT_SLICE,
    NONCONTACT_DIM,
    STATE_DIM,
    build_window_features,
    make_training_samples,
)
from .model import ContactActionTransition

__all__ = [
    "ACTION_DIM",
    "CONTACT_SLICE",
    "NONCONTACT_DIM",
    "STATE_DIM",
    "ContactActionTransition",
    "build_window_features",
    "make_training_samples",
]
