import unittest
import numpy as np
from scripts.material_oracle_probe import analytic_fixture
from scripts.audit_material_sections import analyze, section, graph, cycle_core


class MaterialSectionTests(unittest.TestCase):
    def test_solid_box_is_one_loop(self):
        v,f=analytic_fixture(False)
        r=analyze(v,f,2,.13)
        self.assertEqual(len(r['loops']),1)
        self.assertEqual((r['endpoints'],r['branch_nodes'],r['transverse_crossings']),(0,0,0))

    def test_thick_open_cup_has_two_walls_and_open_top(self):
        v,f=analytic_fixture(True)
        r=analyze(v,f,2,1.)
        self.assertEqual(len(r['loops']),2)
        self.assertEqual(sorted(map(len,r['leaf_pruned_core']['loop_containment'])),[0,1])
        self.assertEqual(r['endpoints'],0)
        center=r['scanlines'][1]
        np.testing.assert_allclose([h['coordinate_m'] for h in center['hits']],[-1,-.8,.8,1])
        np.testing.assert_allclose(center['consecutive_gaps_m'],[.2,1.6,.2])
        self.assertEqual(analyze(v,f,2,2.1)['raw_segments'],0)
        self.assertEqual(len(analyze(v,f,2,.1)['loops']),1)

    def test_reverse_duplicate_does_not_become_thickness(self):
        v,f=analytic_fixture(True)
        r=analyze(v,np.vstack((f,f[:,[0,2,1]])),2,1.)
        self.assertEqual(len(r['loops']),2)
        self.assertEqual(r['duplicate_segments'],r['unique_edges'])
        self.assertEqual(r['duplicate_edges_with_opposed_raw_normals'],r['unique_edges'])

    def test_coplanar_faces_flagged_and_gaps_not_filled(self):
        tri=np.array([[[0.,0.,0.],[1,0,0],[0,1,0]]])
        seg,ids,coplanar=section(tri,2,0.,1e-10)
        self.assertEqual(coplanar,[0])
        self.assertEqual(len(seg),0)
        g=graph(np.array([[[0.,0],[1,0]],[[1,0],[1,1]]]),[1,2],1e-8)
        self.assertEqual(g['endpoints'],2)
        self.assertEqual(len(g['loops']),0)

    def test_closed_ring_with_spur_is_not_called_missing(self):
        s=np.array([[[0.,0],[1,0]],[[1,0],[1,1]],[[1,1],[0,1]],[[0,1],[0,0]],[[1,1],[2,2]]])
        g=graph(s,list(range(5)),1e-8)
        self.assertEqual(len(g['loops']),0)
        core=cycle_core(g,1e-8)
        self.assertEqual(len(core['loops']),1)
        self.assertEqual(core['removed_unique_edges'],1)


if __name__=='__main__':unittest.main()
