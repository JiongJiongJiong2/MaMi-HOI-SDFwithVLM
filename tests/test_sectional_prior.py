import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


from manip.geometry.sectional_prior import (
    build_contact_section_frame,
    evaluate_contact_pair,
    make_box_surface_candidates,
    point_line_distance,
    summarize_results,
    world_to_object,
)


class SectionalPriorGeometryTests(unittest.TestCase):
    def setUp(self):
        self.points, self.normals = make_box_surface_candidates(
            (1.0, 0.75, 0.5),
            steps=9,
        )

    def test_contact_frame_contains_same_height_opposing_point(self):
        source = (1.0, 0.375, 0.25)
        target = (-1.0, -0.375, 0.25)
        frame = build_contact_section_frame(
            source,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            2.0,
        )
        self.assertAlmostEqual(frame.vertical_plane_distance(target), 0.0, places=7)
        self.assertAlmostEqual(frame.horizontal_plane_distance(target), 0.0, places=7)
        self.assertAlmostEqual(frame.chord_distance(target), 0.0, places=7)
        self.assertGreater(frame.inward_travel(target), 0.0)

    def test_contact_above_centre_uses_stable_fallback_axis(self):
        frame = build_contact_section_frame(
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            2.0,
        )
        self.assertAlmostEqual(sum(value * value for value in frame.radial), 1.0, places=7)
        self.assertAlmostEqual(sum(value * value for value in frame.vertical_normal), 1.0, places=7)
        self.assertAlmostEqual(sum(frame.radial[i] * frame.up[i] for i in range(3)), 0.0, places=7)

    def test_world_to_object_matches_row_vector_convention(self):
        rotation = (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        result = world_to_object((1.0, 3.0, 3.0), rotation, (1.0, 2.0, 3.0))
        self.assertEqual(result, (1.0, 0.0, 0.0))

    def test_section_chord_recovers_box_pair_better_than_central_antipode(self):
        row = evaluate_contact_pair(
            sequence_name="box",
            absolute_frame=0,
            source_hand="left",
            source_point=(1.0, 0.375, 0.25),
            target_point=(-1.0, -0.375, 0.25),
            candidates=self.points,
            normals=self.normals,
            centre=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            extent=2.0,
            seed=7,
        )
        self.assertLessEqual(row["section_chord_rank"], row["center_antipode_rank"])
        self.assertEqual(row["section_chord_top5_hit"], 1.0)
        self.assertAlmostEqual(row["target_chord_distance_norm"], 0.0, places=7)
        self.assertEqual(
            (
                row["section_chord_top1_x"],
                row["section_chord_top1_y"],
                row["section_chord_top1_z"],
            ),
            (row["target_x"], row["target_y"], row["target_z"]),
        )

    def test_off_chord_point_has_positive_distance(self):
        distance = point_line_distance(
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )
        self.assertAlmostEqual(distance, 1.0, places=7)

    def test_summary_is_sequence_macro_averaged(self):
        rows = []
        for sequence_name, source, target in (
            ("a", (1.0, 0.0, 0.25), (-1.0, 0.0, 0.25)),
            ("b", (1.0, 0.375, -0.25), (-1.0, -0.375, -0.25)),
        ):
            rows.append(
                evaluate_contact_pair(
                    sequence_name=sequence_name,
                    absolute_frame=0,
                    source_hand="left",
                    source_point=source,
                    target_point=target,
                    candidates=self.points,
                    normals=self.normals,
                    centre=(0.0, 0.0, 0.0),
                    up=(0.0, 0.0, 1.0),
                    extent=2.0,
                    seed=5,
                )
            )
        summary = summarize_results(rows)
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["sequence_count"], 2)
        self.assertIn("section_chord", summary["methods"])


class SectionalPriorCLITests(unittest.TestCase):
    def test_synthetic_cli_writes_complete_artifacts_without_numpy(self):
        repository_root = Path(__file__).resolve().parents[1]
        script = repository_root / "scripts" / "run_sectional_prior_diagnostic.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "result"
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
            self.assertTrue((output_dir / "samples.csv").exists())
            self.assertTrue((output_dir / "section_frames.jsonl").exists())
            summary_path = output_dir / "summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "synthetic")
            self.assertGreater(summary["sample_count"], 0)
            self.assertIn(summary["diagnostic_gate"]["status"], {"PASS", "FAIL"})

    def test_cli_rejects_odd_sample_limit_to_preserve_hand_pairs(self):
        repository_root = Path(__file__).resolve().parents[1]
        script = repository_root / "scripts" / "run_sectional_prior_diagnostic.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--synthetic", "--max_samples", "3"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--max_samples must be even", completed.stderr)


if __name__ == "__main__":
    unittest.main()
