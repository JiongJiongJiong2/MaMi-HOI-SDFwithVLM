import unittest
import numpy as np
from scripts.material_oracle_probe import analytic_fixture
from scripts.probe_plasticbox_rim import certify_segment


class RimPathTests(unittest.TestCase):
    def test_cup_cavity_has_clear_exit_above_open_top(self):
        v,f=analytic_fixture(True)
        r=certify_segment([0,0,1],[0,0,2.5],v[f],1e-7)
        self.assertEqual(r['status'],'CLEAR_TO_AVAILABLE_TRIANGLES')
        self.assertGreater(r['certified_lower_bound_m'],0)

    def test_solid_material_cannot_exit_without_crossing(self):
        v,f=analytic_fixture(False)
        r=certify_segment([0,0,0],[0,0,2],v[f],1e-7,max_queries=128)
        self.assertNotEqual(r['status'],'CLEAR_TO_AVAILABLE_TRIANGLES')

    def test_crossing_between_clear_endpoints_is_not_missed(self):
        v,f=analytic_fixture(False)
        r=certify_segment([-2,0,0],[2,0,0],v[f],1e-7,max_queries=128)
        self.assertNotEqual(r['status'],'CLEAR_TO_AVAILABLE_TRIANGLES')


if __name__=='__main__':unittest.main()
