import unittest
import numpy as np

from LUTridiagonalVectores import factorizacionLU, resolverSistemaLU, resolverSistemaConLU

class LUTridiagonalVectoresTestCase(unittest.TestCase):

    def test_ok_1(self):
        m = np.array([
        [-1, -9,  0,  0, 0],
        [-6, -4,  2,  0,  0],
        [ 0, -9,  3, -5,  0],
        [0, 0,  6,  5,  7],
        [0, 0,  0, -9,  2]])
        b = np.array([-46, -16,  10, -73,  54]).T
        x = np.linalg.solve(m,b)

        v = factorizacionLU(m.tolist())
        sol = resolverSistemaLU(v[0], v[1], v[2], b.tolist())
        np.testing.assert_allclose(sol, x)

    def test_error_2(self):
        m = np.array([
        [0, 5, 0, 0],
        [2, 1, 4, 0],
        [0, 5, 2, 3],
        [0, 0, 6, 1]])
        b = np.array([15, 21, -1, 16]).T
        x = np.linalg.solve(m,b)

        v = factorizacionLU(m.tolist())
        sol = resolverSistemaLU(v[0], v[1], v[2], b.tolist())
        np.testing.assert_allclose(sol, x)
    
    def test_error_3(self):
        m = np.array([
        [2, 4, 0, 0, 0],
        [6, 12, 5, 0, 0],
        [0, 3, 3, 1, 0],
        [0, 0, 1, 2, -4],
        [0, 0, 0, 8, -6]])
        b = np.array([-3., 3., -7.06, 0., 1.]).T 
        x = np.linalg.solve(m,b)

        v = factorizacionLU(m.tolist())
        sol = resolverSistemaLU(v[0], v[1], v[2], b.tolist())
        np.testing.assert_allclose(sol, x)

    def test_ok_4(self):
        m = np.array([
        [2, 1, 0, 0],
        [4, 3, 3, 0],
        [ 0, 7, 9, 5],
        [0, 0, 9, 8]])
        b = np.array([-16,  10, -73,  54]).T
        x = np.linalg.solve(m,b)

        v = factorizacionLU(m.tolist())
        sol = resolverSistemaLU(v[0], v[1], v[2], b.tolist())
        np.testing.assert_allclose(sol, x)
    
    def test_ok_5(self):
        m = np.array([
        [2, 1, 0, 0, 0],
        [4, 9, 3, 0, 0],
        [0, 7, 9, 5, 0],
        [0, 0, 9, 0, 4],
        [0, 0, 0, 1, 9]])
        b = np.array([-16,  10, -73, 10, 54]).T
        x = np.linalg.solve(m,b)

        v = factorizacionLU(m.tolist())
        sol = resolverSistemaLU(v[0], v[1], v[2], b.tolist())
        np.testing.assert_allclose(sol, x)
 
if __name__ == '__main__':
    unittest.main()