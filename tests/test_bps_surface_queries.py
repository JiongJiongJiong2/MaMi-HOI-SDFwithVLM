"""Analytic checks for the standalone CPU geometry diagnostic."""
import unittest
import numpy as np
from scripts.audit_bps_surface_queries import (
    nearest_triangles, nearest_points, world_to_canonical, compare_queries,
)


class GeometryQueryTests(unittest.TestCase):
    def test_face_edge_vertex_and_degenerate(self):
        tri = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]])
        q = np.array([[.2, .3, 2], [.7, .7, 0], [-1, -1, 0]])
        d, cp = nearest_triangles(q, tri, 1, 1)
        np.testing.assert_allclose(cp, [[.2, .3, 0], [.5, .5, 0], [0, 0, 0]], atol=1e-14)
        np.testing.assert_allclose(d, [2, np.sqrt(.08), np.sqrt(2)])
        degenerate = np.array([[[0., 0, 0], [1, 0, 0], [1, 0, 0]]])
        d, cp = nearest_triangles([[.5, 1, 0]], degenerate)
        np.testing.assert_allclose(d, [1])
        np.testing.assert_allclose(cp, [[.5, 0, 0]])

    def test_cube_analytic_chunk_and_winding(self):
        v = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        faces = []
        for axis in range(3):
            for side in (-1, 1):
                ids = np.flatnonzero(v[:, axis] == side)
                faces.extend([[ids[0], ids[1], ids[3]], [ids[0], ids[3], ids[2]]])
        tri = v[np.array(faces)]
        q = np.random.default_rng(73).uniform(-2, 2, (47, 3))
        outside = np.maximum(abs(q) - 1, 0)
        expected = np.where((abs(q) <= 1).all(1), (1-abs(q)).min(1), np.linalg.norm(outside, axis=1))
        d, cp = nearest_triangles(q, tri, 3, 2)
        np.testing.assert_allclose(d, expected, atol=1e-13)
        d2, cp2 = nearest_triangles(q, tri[:, ::-1], 16, 1024)
        np.testing.assert_allclose(d2, d, atol=1e-13)
        np.testing.assert_allclose(cp2, cp, atol=1e-13)

    def test_rigid_transform_and_point_subset(self):
        r = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        com = np.array([4., -3, 2])
        q = np.array([[.2, .3, 1], [2, 2, 2.]])
        tri = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]])
        world = q @ r.T + com
        np.testing.assert_allclose(world_to_canonical(world, r, com), q, atol=1e-14)
        d, cp = nearest_triangles(q, tri)
        dw, cpw = nearest_triangles(world, tri @ r.T + com)
        np.testing.assert_allclose(dw, d, atol=1e-14)
        np.testing.assert_allclose(cpw, cp @ r.T + com, atol=1e-14)
        vd, vcp, _ = nearest_points(q, tri.reshape(-1, 3), 1, 1)
        self.assertTrue((vd >= d).all())
        err, angle = compare_queries(q, vd, vcp, d, cp, 1.)
        self.assertTrue(np.isfinite(angle).all())
        self.assertTrue((err >= 0).all())

    def test_surface_direction_is_undefined(self):
        q = np.array([[.2, .3, 0.]])
        err, angle = compare_queries(q, np.array([1.]), np.array([[0., 0, 1]]), np.zeros(1), q, 1.)
        self.assertTrue(np.isnan(angle[0]))
        with self.assertRaises(ValueError):
            world_to_canonical(q, np.eye(3)*2, np.zeros(3))


if __name__ == '__main__':
    unittest.main()
