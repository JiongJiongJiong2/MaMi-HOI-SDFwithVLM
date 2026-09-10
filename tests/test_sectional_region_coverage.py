import unittest
import numpy as np
from scripts.audit_sectional_region_coverage import diverse


class RegionCoverageTests(unittest.TestCase):
    def test_preserves_top_one_and_removes_nearby_redundancy(self):
        p=np.array([[0.,0,0],[.01,0,0],[.09,0,0],[.2,0,0]])
        self.assertEqual(diverse([0,1,2,3],p,.08,3),[0,2,3])
        self.assertEqual(diverse([3,2,1,0],p,.08,2),[3,2])

    def test_insufficient_regions_are_not_backfilled(self):
        p=np.array([[0.,0,0],[.01,0,0]])
        self.assertEqual(diverse([0,1],p,.08,10),[0])
        self.assertEqual(diverse([0,1],p*5,.08*5,10),[0])


if __name__=='__main__':unittest.main()
