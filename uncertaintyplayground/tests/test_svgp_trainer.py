import unittest
import numpy as np
from uncertaintyplayground.trainers.svgp_trainer import SparseGPTrainer

class TestSparseGPTrainer(unittest.TestCase):
    """
    Unit test suite for the SparseGPTrainer class.

    This class contains a series of methods to test the functionalities of SparseGPTrainer, including training and making predictions with uncertainty.

    Attributes:
        X (np.array): An array of feature vectors for the training data.
        y (np.array): An array of target values for the training data.
        trainer (SparseGPTrainer): The SparseGPTrainer instance to test.
    """

    def setUp(self):
        """
        Setup function that runs before each test method.

        This method generates random data for the tests and initializes an instance of SparseGPTrainer.
        """
        self.X = np.random.rand(100, 20)
        self.y = np.random.rand(100)
        self.trainer = SparseGPTrainer(self.X, self.y, num_inducing_points=20, num_epochs=10, batch_size=20, lr=0.2, patience=3)

    def test_train(self):
        """
        Tests the train method of the SparseGPTrainer class.

        This test trains the model and checks that it has non-zero parameters afterward, verifying that training has indeed happened.
        """
        self.trainer.train()
        self.assertTrue(any(p.detach().numpy().any() for p in self.trainer.model.parameters()))

    def test_predict_with_uncertainty(self):
        """
        Tests the predict_with_uncertainty method of the SparseGPTrainer class.

        This test trains the model and then makes a prediction with uncertainty. It checks that the predictions and uncertainties have the correct shape.
        """
        self.trainer.train()
        y_pred, y_var = self.trainer.predict_with_uncertainty(self.X)
        self.assertEqual(y_pred.shape, (self.X.shape[0],))
        self.assertEqual(y_var.shape, (self.X.shape[0],))

if __name__ == "__main__":
    unittest.main()
