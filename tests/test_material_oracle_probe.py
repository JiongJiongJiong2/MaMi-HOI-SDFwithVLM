import unittest
import numpy as np
from scripts.material_oracle_probe import (
    analytic_fixture,analytic_truth,run_fixtures,winding,parity,
    conservative_candidate,topology,classification_metrics,ray_hits,
)


class MaterialOracleTests(unittest.TestCase):
    def test_analytic_box_and_open_cup(self):
        r=run_fixtures()
        self.assertTrue(r['open_thick_cup']['no_cap_at_cavity'])
        self.assertEqual(r['closed_box']['occupancy_accuracy'],1.)

    def test_cup_walls_bottom_cavity_opening(self):
        v,f=analytic_fixture(True)
        p=np.array([[.9,0,1],[0,0,.1],[0,0,1],[0,0,2.2],[1.2,0,1],[.9,0,2.01]])
        labels,d=analytic_truth(p,True)
        np.testing.assert_array_equal(labels,[1,1,0,0,0,0])
        np.testing.assert_allclose(d,[.1,.1,.8,np.sqrt(.8**2+.2**2),.2,.01])
        np.testing.assert_array_equal(winding(p,v[f])>.5,labels)
        info,_=topology(v,f)
        self.assertEqual(info['boundary_edges'],0)
        self.assertEqual(len(ray_hits([0,0,1],[0,0,1],v[f],2.)[0]),0)

    def test_opposite_duplicates_and_candidate(self):
        v,f=analytic_fixture(False)
        duplicated=np.vstack([f,f[:,[0,2,1]]])
        p=np.array([[0.,0,0],[2,0,0]])
        np.testing.assert_allclose(winding(p,v[duplicated]),0,atol=1e-12)
        unique,naive=parity(p,v[duplicated],2.)
        np.testing.assert_array_equal(unique[0],1)
        np.testing.assert_array_equal(naive[0],0)
        cv,cf,info,_=conservative_candidate(v,duplicated)
        self.assertEqual(info['boundary_edges'],0)
        np.testing.assert_array_equal(winding(p,cv[cf])>.5,[True,False])

    def test_orientation_and_open_failure_are_visible(self):
        v,f=analytic_fixture(False)
        self.assertLess(winding([[0,0,0]],v[f[:,[0,2,1]]])[0],-.99)
        cv,cf,info,_=conservative_candidate(v,f[:-2])
        self.assertGreater(info['boundary_edges'],0)
        self.assertEqual(info['filled_holes'],0)

    def test_unknown_is_not_true_or_false_and_empty_accuracy_is_null(self):
        rows=[dict(id='a',label='unknown'),dict(id='b',label='free')]
        m=classification_metrics(np.array([1,0]),rows)
        self.assertIsNone(m['material']['accuracy'])
        self.assertEqual(m['unknown_label_abstention_fraction'],0.)
        self.assertEqual(m['free']['accuracy'],1.)
        abstained=classification_metrics(np.array([-1,-1]),rows)
        self.assertEqual(abstained['misclassified_ids'],[])
        self.assertEqual(abstained['abstained_known_ids'],['b'])

    def test_rigid_and_scale_invariance(self):
        v,f=analytic_fixture(True)
        p=np.array([[.9,0,1],[0,0,1],[2.,0,1]])
        r=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        cv=(v@r.T)*.03+[4,-2,1];cp=(p@r.T)*.03+[4,-2,1]
        np.testing.assert_allclose(winding(cp,cv[f]),winding(p,v[f]),atol=1e-12)


if __name__=='__main__':unittest.main()
