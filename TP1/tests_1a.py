import unittest
import numpy as np

from eliminacionGaussiana import eliminacionGaussiana_SP

class EliminacionGaussianaTestCase(unittest.TestCase):

    def test_excepcion_1(self):
        a = [[2, 1, -1, 3],
        [-2, 0, 0, 0],
        [4, 1, -2, 4],         
        [-6, -1, 2, -3]]
        b = [13, -2, 24]
        eliminacionGaussiana_SP(a, b)
    
    def test_ok_2(self):
        a = [[2, 1, -1, 3],
        [-2, 0, 0, 0],
        [4, 1, -2, 4],         
        [-6, -1, 2, -3]]
        b = [13, -2, 24, -10]
        sol = eliminacionGaussiana_SP(a, b)
        np.testing.assert_allclose(sol, [1,-30,7,16])

    def test_ok_3(self):
        a = np.array([[-1, -9,  6,  3, -2],
        [-6, -4,  2,  8,  5],
        [ 5, -9,  3, -5,  6],
        [-6, -9,  6,  5,  7],
        [-1, -8,  0, -9,  2]])
        b = np.array([-22, -125, -39, -124, 13])
        x = np.linalg.solve(a,b)
       
        sol = eliminacionGaussiana_SP(a.tolist(),b.tolist())
        np.testing.assert_allclose(sol, x.tolist())

    def test_error_4(self):
        a = [[0, -2, -1],
        [2, 3, 1],
        [3, 1, -1]]
        b = [-14, 1, 1]
        sol = eliminacionGaussiana_SP(a, b)
        np.testing.assert_allclose(sol, [-18,23,-32])

    def test_error_5(self):
        a = [[1,2,3],
        [3,3,3],
        [4,8,12]]
        b = [1,1,1]
        eliminacionGaussiana_SP(a,b)
    
    def test_ok_6(self):
        m = np.array([
        [-1, -9,  0,  0, 0],
        [-6, -4,  2,  0,  0],
        [ 0, -9,  3, -5,  0],
        [0, 0,  6,  5,  7],
        [0, 0,  0, -9,  2]])
        b = np.array([-46, -16, 10, -73, 54])
        x = np.linalg.solve(m,b)
        
        sol = eliminacionGaussiana_SP(m.tolist(), b.tolist())
        np.testing.assert_allclose(sol, x.tolist())
    
if __name__ == '__main__':
    unittest.main()

