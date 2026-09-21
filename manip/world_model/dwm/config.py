"""Frozen configuration for the DWM MuJoCo pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PHYSICS_TIMESTEP = 0.002
PHYSICS_STEPS_PER_CONTROL = 15
CONTROL_PERIOD = PHYSICS_TIMESTEP * PHYSICS_STEPS_PER_CONTROL
ACTION_DIM = 51
RESET_SEED = 20260922
RESETS_PER_OBJECT = 120

PD_GAINS = {
    "wrist_translation": (1000.0, 100.0),
    "wrist_rotation": (50.0, 5.0),
    "finger": (5.0, 0.5),
}

SHAPE_SIZES = {
    "sphere": (0.040,),
    "box": (0.040, 0.030, 0.025),
    "cylinder": (0.035, 0.035),
}

VARIANTS = (
    {"mass": 0.10, "friction": 0.45, "split": "train"},
    {"mass": 0.25, "friction": 0.65, "split": "train"},
    {"mass": 0.35, "friction": 0.80, "split": "val"},
    {"mass": 0.55, "friction": 1.00, "split": "test"},
)


@dataclass(frozen=True)
class ObjectConfig:
    object_id: str
    shape: str
    variant: int
    mass: float
    friction: float
    split: str
    size: tuple[float, ...]

    @property
    def rest_height(self):
        if self.shape == "sphere":
            return self.size[0]
        if self.shape == "box":
            return self.size[2]
        if self.shape == "cylinder":
            return self.size[1]
        raise ValueError(f"Unsupported shape: {self.shape}")

    @property
    def contact_wrist_offset(self):
        return {
            "sphere": 0.050,
            "box": 0.035,
            "cylinder": 0.040,
        }[self.shape]

    @property
    def near_contact_wrist_offset(self):
        return self.contact_wrist_offset + 0.015

    def geom_xml(self):
        size = " ".join(f"{value:.6f}" for value in self.size)
        if self.shape == "sphere":
            geom_type = "sphere"
        elif self.shape == "box":
            geom_type = "box"
        elif self.shape == "cylinder":
            geom_type = "cylinder"
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")
        rgba = {
            "sphere": "0.18 0.52 0.82 1",
            "box": "0.85 0.42 0.20 1",
            "cylinder": "0.24 0.68 0.38 1",
        }[self.shape]
        return (
            f'<geom name="dwm_object_geom" type="{geom_type}" '
            f'size="{size}" mass="{self.mass:.6f}" '
            f'friction="{self.friction:.6f} {self.friction:.6f} 0.001" '
            f'rgba="{rgba}" />'
        )


def _build_object_configs():
    rows = []
    for shape, size in SHAPE_SIZES.items():
        for variant, values in enumerate(VARIANTS):
            rows.append(ObjectConfig(
                object_id=f"{shape}_v{variant}",
                shape=shape,
                variant=variant,
                mass=values["mass"],
                friction=values["friction"],
                split=values["split"],
                size=tuple(size),
            ))
    return tuple(rows)


OBJECT_CONFIGS = _build_object_configs()
OBJECT_CONFIG_BY_ID = {
    config.object_id: config for config in OBJECT_CONFIGS
}


def object_configs_for_split(split):
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unsupported split: {split}")
    return tuple(
        config for config in OBJECT_CONFIGS
        if config.split == split
    )


def right_hand_xml_path():
    return (
        Path(__file__).resolve().parents[3]
        / "third_party"
        / "handx_smplx_hand"
        / "grab_s1_rhand.xml"
    )
