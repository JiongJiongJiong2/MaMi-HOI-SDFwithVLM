import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from manip.geometry.paired_region import (
    closest_triangle_points,
    diverse_order,
    evaluate_projected_contact,
    sample_surface_candidates,
)


def _cube():
    vertices = np.asarray(
        [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ],
        dtype=np.int64,
    )
    return vertices, faces


class PairedRegionGeometryTests(unittest.TestCase):
    def test_triangle_projection_returns_face_and_normal(self):
        triangles = np.asarray(
            [[(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)]],
            dtype=np.float64,
        )
        distances, closest, face_ids, normals = closest_triangle_points(
            np.asarray([(0.0, 0.0, 2.0)]),
            triangles,
        )
        self.assertAlmostEqual(float(distances[0]), 2.0)
        np.testing.assert_allclose(closest[0], (0.0, 0.0, 0.0))
        self.assertEqual(int(face_ids[0]), 0)
        np.testing.assert_allclose(np.abs(normals[0]), (0.0, 0.0, 1.0))

    def test_diverse_order_requires_separation(self):
        points = np.asarray(
            [(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.2, 0.0, 0.0)],
            dtype=np.float64,
        )
        self.assertEqual(diverse_order([0, 1, 2], points, 0.08, 2), [0, 2])

    def test_independent_target_projection_controls_pool_and_oracle(self):
        vertices, faces = _cube()
        points, normals = sample_surface_candidates(
            vertices,
            faces,
            count=512,
            seed=7,
        )
        result = evaluate_projected_contact(
            sequence_name="box",
            absolute_frame=0,
            source_hand="left",
            source_point=np.asarray((1.0, 0.2, 0.2)),
            source_normal=np.asarray((1.0, 0.0, 0.0)),
            target_point=np.asarray((-1.0, -0.2, 0.2)),
            target_normal=np.asarray((-1.0, 0.0, 0.0)),
            candidates=points,
            normals=normals,
            centre=np.asarray((0.0, 0.0, 0.0)),
            up=np.asarray((0.0, 0.0, 1.0)),
            extent=2.0,
            seed=3,
        )
        self.assertIn("pool_recall", result)
        self.assertIn("eligible_pool_recall", result)
        self.assertIn("oracle_top5_hit", result)
        self.assertIn("section_chord_normal_top5_hit", result)


class BPR01CLITests(unittest.TestCase):
    def test_synthetic_cli_writes_complete_artifacts(self):
        repository_root = Path(__file__).resolve().parents[1]
        script = (
            repository_root
            / "scripts"
            / "run_bpr01_paired_region_diagnostic.py"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "bpr01"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--synthetic",
                    "--output_dir",
                    str(output_dir),
                ],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertTrue((output_dir / "samples.csv").is_file())
            self.assertTrue((output_dir / "protocol.json").is_file())
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["experiment"], "BPR-01")
            self.assertEqual(summary["mode"], "synthetic")
            self.assertIn("clean", summary["variants"])
            self.assertIn("diagnostic_gate", summary)


if __name__ == "__main__":
    unittest.main()
