import smithy
import unittest
import numpy as np

class TestOpsPysmithy (unittest.TestCase):



    def test_relu(self):
        self.assertEqual(smithy.relu(3), 3.0)
        self.assertEqual(smithy.relu(-3), 0.0)
 
    def test_cosh(self):
        self.assertEqual(smithy.cosh(0),1.0)
        self.assertAlmostEqual(smithy.cosh(-100), 1.3440585709080678e+43, places=5)
        self.assertAlmostEqual(smithy.cosh(100), 1.3440585709080678e+43, places=5)

    def test_tanh(self):
        self.assertEqual(smithy.tanh(0), 0.0)
        self.assertAlmostEqual(smithy.tanh(1), 0.76159, places=5)
        self.assertAlmostEqual(smithy.tanh(-1), -0.76159, places=5)
    
    def test_matrix_multiply(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])

        result = smithy.dot_matrice(a, b)

        expected_result = np.array([[19, 22], [43, 50]])

        np.testing.assert_array_equal(result, expected_result)

    def test_matrix_multiply_non_square(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8], [9, 10], [11, 12]])

        result = smithy.dot_matrice(a, b)

        expected_result = np.array([[58, 64], [139, 154]])

        np.testing.assert_array_equal(result, expected_result)


    


if __name__ == '__main__':
        unittest.main(verbosity=2)



