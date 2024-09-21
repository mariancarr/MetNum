import unittest
import numpy as np

from eliminacionGaussianaPivot import eliminacionGaussiana_PP

epsilon = 1e-8
class EliminacionGaussianaTestCase(unittest.TestCase):

    def test_excepcion_1(self):
        a = [[2, 1, -1, 3],
        [-2, 0, 0, 0],
        [4, 1, -2, 4],         
        [-6, -1, 2, -3]]
        b = [13, -2, 24]
        print("\ntest 1")
        eliminacionGaussiana_PP(a, b, epsilon)
    
    def test_ok_2(self):
        a = np.array([[2, 1, -1, 3],
        [-2, 0, 0, 0],
        [4, 1, -2, 4],
        [-6, -1, 2, -3]])
        b = np.array([13, -2, 24, -10])
        x = np.linalg.solve(a,b)

        print("\ntest 2")
        sol = eliminacionGaussiana_PP(a.tolist(), b.tolist(),epsilon)      
        np.testing.assert_allclose(sol, x)

    def test_ok_3(self):
        a = np.array([[-1, -9,  6,  3, -2],
        [-6, -4,  2,  8,  5],
        [ 5, -9,  3, -5,  6],
        [-6, -9,  6,  5,  7],
        [-1, -8,  0, -9,  2]])
        b = np.array([1, 5, 5, -8, -9])
        x = np.linalg.solve(a,b)
        
        print("\ntest 3")
        sol = eliminacionGaussiana_PP(a.tolist(),b.tolist(),epsilon)
        np.testing.assert_allclose(sol,x)

    def test_ok_4(self):
        a = np.array([[0, -2, -1],
        [2, 3, 1],
        [3, 1, -1]])
        b = np.array([-14, 1, 1])
        x = np.linalg.solve(a,b)
        
        print("\ntest 4")
        sol = eliminacionGaussiana_PP(a.tolist(), b.tolist(),epsilon)
        np.testing.assert_allclose(sol, x)

    def test_error_5(self):
        a = [[1,2,3],
        [3,3,3],
        [4,8,12]]
        b = [1,1,1]
       
        print("\ntest 5")
        eliminacionGaussiana_PP(a,b,epsilon)
  
    def test_ok_6(self):
        a = np.array([
        [-1, -9,  0,  0, 0],
        [-6, -4,  2,  0,  0],
        [ 0, -9,  3, -5,  0],
        [0, 0,  6,  5,  7],
        [0, 0,  0, -9,  2]])
        b = np.array([1, 5, 5, -8, -9])
        x = np.linalg.solve(a,b)
        
        print("\ntest 6")
        sol = eliminacionGaussiana_PP(a.tolist(), b.tolist(),epsilon)
        np.testing.assert_allclose(sol,x)

    def test_ok_7(self):
        a = np.array([
        [0, 5, 0, 0],
        [2, 1, 4, 0],
        [0, 5, 2, 3],
        [0, 0, 6, 1]])
        b = np.array([1, 3, 4, -8])
        x = np.linalg.solve(a,b) 

        print("\ntest 7")
        sol = eliminacionGaussiana_PP(a.tolist(), b.tolist(),epsilon)
        np.testing.assert_allclose(sol,x)

    def test_ok_8(self):
        a = np.array([
        [2, 4, 0, 0, 0],
        [6, 12, 5, 0, 0],
        [0, 3, 3, 1, 0],
        [0, 0, 1, 2, -4],
        [0, 0, 0, 8, -6]])
        b = np.array([8.62, -5.06, 2.4, 0.92, 1.06]) 
        x = np.linalg.solve(a,b)

        print("\ntest 8")
        sol = eliminacionGaussiana_PP(a.tolist(), b.tolist(),epsilon)  
        np.testing.assert_allclose(sol,x)

if __name__ == '__main__':
    unittest.main()