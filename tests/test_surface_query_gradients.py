"""Independent analytic acceptance and no-overwrite CLI checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.probe_surface_query_gradients import fixtures


class SurfaceQueryGradientTests(unittest.TestCase):
    def test_face_edge_and_medial_fixture(self):
        report = fixtures()
        self.assertTrue(all(p['status'] == 'PASS_SAMPLED_DERIVATIVE' for p in report['smooth']))
        self.assertEqual(report['expected_medial_nondifferentiability']['status'],
                         'REVIEW_NONSMOOTH_OR_ERROR')

    def test_output_guards_preserve_files(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/probe_surface_query_gradients.py'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            data.mkdir()
            existing = root / 'existing'
            existing.mkdir()
            sentinel = existing / 'summary.json'
            sentinel.write_text(json.dumps({'preserve': True}))
            before = sentinel.read_bytes()
            for output in (existing, data, data / 'new_child'):
                result = subprocess.run([sys.executable, '-B', str(script),
                    '--data_root_folder', str(data), '--probe_manifest', str(root / 'absent.json'),
                    '--output_dir', str(output)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('absent.json', result.stderr)
                self.assertIn('FileExistsError' if output == existing else 'ValueError', result.stderr)
            self.assertEqual(sentinel.read_bytes(), before)
            self.assertFalse((data / 'new_child').exists())


if __name__ == '__main__':
    unittest.main()
