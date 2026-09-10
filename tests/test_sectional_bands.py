import unittest
import numpy as np
from scripts.audit_sectional_bands import band_orders
from manip.geometry.sectional_prior import build_contact_section_frame


class SectionBandTests(unittest.TestCase):
    def test_union_accepts_either_plane_and_excludes_source(self):
        source=np.array([.5,0.,0.])
        p=np.array([[-.5,.3,0],[-.5,0,.3],[-.5,.3,.3],source])
        frame=build_contact_section_frame(source,[0,0,0],[0,0,1],1.)
        orders=band_orders(p,source,frame,1.)
        self.assertEqual(orders['union_band'],[0,1,2])
        self.assertEqual(orders['horizontal_band'][0],0)
        self.assertEqual(orders['vertical_band'][0],1)
        self.assertTrue(all(3 not in o for o in orders.values()))

    def test_rigid_scale_preserves_ranking(self):
        source=np.array([.5,0.,0.]);p=np.array([[-.5,.3,0],[-.5,0,.3],[-.5,.3,.3]])
        frame=build_contact_section_frame(source,[0,0,0],[0,0,1],1.)
        rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);shift=np.array([1.,2.,3.]);scale=3.
        other=build_contact_section_frame(source@rotation*scale+shift,shift,np.array([0,0,1])@rotation,scale)
        self.assertEqual(band_orders(p,source,frame,1.),band_orders(p@rotation*scale+shift,source@rotation*scale+shift,other,scale))


if __name__=='__main__':unittest.main()
