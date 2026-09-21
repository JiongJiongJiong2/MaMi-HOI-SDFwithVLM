"""Build the DWM MuJoCo model from the frozen right-hand asset."""

from __future__ import annotations

from .config import (
    CONTROL_PERIOD,
    ObjectConfig,
    right_hand_xml_path,
)


def build_model_xml(config: ObjectConfig, include_floor: bool = True):
    if not isinstance(config, ObjectConfig):
        raise TypeError("config must be an ObjectConfig")
    path = right_hand_xml_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    xml = path.read_text(encoding="utf-8")

    inertial_bodies = {
        '<body name="Pelvis" pos="0 0 0">': (
            '<body name="Pelvis" pos="0 0 0">\n'
            '      <inertial pos="0 0 0" mass="0.100000" '
            'diaginertia="0.001000 0.001000 0.001000" />'
        ),
        '<body name="R_Wrist_s" pos="0 0 0">': (
            '<body name="R_Wrist_s" pos="0 0 0">\n'
            '        <inertial pos="0 0 0" mass="0.050000" '
            'diaginertia="0.000100 0.000100 0.000100" />'
        ),
    }
    for marker, replacement in inertial_bodies.items():
        if marker not in xml:
            raise ValueError(f"right-hand MJCF missing body marker: {marker}")
        xml = xml.replace(marker, replacement, 1)

    default_geom = (
        '<geom conaffinity="1" condim="3" contype="7" '
        'margin="0.001" rgba="0.8 0.6 .4 1" />'
    )
    if default_geom not in xml:
        raise ValueError("right-hand MJCF default geom block changed")
    xml = xml.replace(
        default_geom,
        (
            '<geom conaffinity="1" condim="3" contype="7" '
            'margin="0.001" rgba="0.8 0.6 .4 1" '
            f'friction="{config.friction:.6f} '
            f'{config.friction:.6f} 0.001" />'
        ),
        1,
    )

    object_body = (
        f'<body name="dwm_object" pos="0 0 {config.rest_height:.6f}">\n'
        '  <freejoint name="dwm_object_free" />\n'
        f'  {config.geom_xml()}\n'
        '</body>\n'
    )
    if "</worldbody>" not in xml:
        raise ValueError("right-hand MJCF is missing worldbody close")
    xml = xml.replace("</worldbody>", object_body + "  </worldbody>", 1)

    if '<option ' not in xml:
        xml = xml.replace(
            '<compiler coordinate="local" />',
            (
                '<compiler coordinate="local" />\n'
                '<option timestep="0.002" integrator="implicitfast" />'
            ),
            1,
        )

    if not include_floor:
        xml = xml.replace(
            '<geom conaffinity="1" condim="3" name="floor"',
            '<geom contype="0" conaffinity="0" name="floor"',
            1,
        )

    return xml
