import sys
import os
sys.path.insert(0, os.path.abspath("C:/Users/Orefice/OneDrive/Bureau/IT/Pysmithy"))

import unittest
from py_impl.model.linear import LinearRegression
import numpy as np



class TestImplementationPythonLinearModel(unittest.TestCase):
    

    def setUp(self):
        """Initialisation des données pour les tests."""
        np.random.seed(42)  # Fixer un seed pour des résultats reproductibles
        self.X_train = np.random.rand(100, 2)  # 100 exemples, 2 features
        self.y_train = (np.random.rand(100, 1) > 0.5).astype(int)  # Labels binaires
        self.X_test = np.random.rand(10, 2)  # 10 nouveaux exemples
        self.model = LinearRegression()
    
    
    def test_training_shapes(self):
        """Vérifie que les poids et le biais ont les bonnes dimensions après entraînement."""
        self.model.fit(self.X_train, self.y_train, learning_rate=0.1, epochs=10)
        self.assertEqual(self.model.weights.shape, (2, 1))  # 2 features, 1 poids par feature
        self.assertTrue(isinstance(self.model.bias, (int, float)))

    
    def test_backward_propagation(self):
        """Test de la méthode de propagation arrière."""
        
        # Modèle avec propagation arrière
        self.model = LinearRegression(resolution=1)
        self.model.fit(self.X_train, self.y_train, learning_rate=0.6, epochs=1000)
        result_backward_prop = self.model.weights  # Récupère les poids après fit
        print("Backward Propagation")
        print(self.model.predict(self.X_test))  # Prédictions sur X_test
        print(f"Weights: {self.model.weights}, Bias: {self.model.bias}")

        # Modèle avec équation normale
        self.model = LinearRegression(resolution=-1)
        self.model.fit(self.X_train, self.y_train)
        result_normal = self.model.weights  # Récupère les poids après fit
        print("Normal Equation")
        print(self.model.predict(self.X_test))  # Prédictions sur X_test
        print(f"Weights: {self.model.weights}, Bias: {self.model.bias}")
        
        # Vérification si les poids sont assez proches
        np.testing.assert_almost_equal(result_backward_prop, result_normal, decimal=5)


        





if __name__ == '__main__':
        unittest.main(verbosity=2)