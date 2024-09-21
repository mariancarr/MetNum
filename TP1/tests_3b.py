import unittest
import numpy as np

from eliminacionGaussianaTridiagonalVectores import eliminacionGaussianaTridigonal

class TridiagonalVectoresTestCase(unittest.TestCase):

    def test_ok_1(self):
        m = np.array([
        [-1, -9,  0,  0, 0],
        [-6, -4,  2,  0,  0],
        [ 0, -9,  3, -5,  0],
        [0, 0,  6,  5,  7],
        [0, 0,  0, -9,  2]])
        b = np.array([-46, -16,  10, -73,  54])
        x = np.linalg.solve(m,b)
      
        sol = eliminacionGaussianaTridigonal(m.tolist(), b.tolist())
        np.testing.assert_allclose(sol, x)

    def test_ok_2(self):
        m = np.array([
        [0, 5, 0, 0],
        [2, 1, 4, 0],
        [0, 5, 2, 3],
        [0, 0, 6, 1]])
        b = np.array([1, 3, 4, -8])
        x = np.linalg.solve(m,b)

        sol = eliminacionGaussianaTridigonal(m.tolist(), b.tolist())
        np.testing.assert_allclose(sol, x)
    
    def test_ok_3(self):
        m = np.array([
        [2, 4, 0, 0, 0],
        [6, 12, 5, 0, 0],
        [0, 3, 3, 1, 0],
        [0, 0, 1, 2, -4],
        [0, 0, 0, 8, -6]])
        b = np.array([8.62, -5.06, 2.4, 0.92, 1.06])
        x = np.linalg.solve(m,b)

        sol = eliminacionGaussianaTridigonal(m.tolist(), b.tolist())  
        np.testing.assert_allclose(sol, x)

    def test_ok_4(self):
        m = np.array([
        [2, 1, 0, 0],
        [4, 3, 3, 0],
        [ 0, 7, 9, 5],
        [0, 0, 9, 8]])
        b = np.array([-16,  10, -73,  54]).T
        x = np.linalg.solve(m,b)

        sol = eliminacionGaussianaTridigonal(m.tolist(), b.tolist())  
        np.testing.assert_allclose(sol, x)

    def test_ok_5(self):
        m = np.array([
        [2, 1, 0, 0, 0],
        [4, 2, 3, 0, 0],
        [0, 7, 9, 5, 0],
        [0, 0, 9, 0, 4],
        [0, 0, 0, 1, 9]])
        b = np.array([-16,  10, -73, 10, 54]).T
        x = np.linalg.solve(m,b)

        sol = eliminacionGaussianaTridigonal(m, b)
        np.testing.assert_allclose(sol, x)

if __name__ == '__main__':
    unittest.main()
