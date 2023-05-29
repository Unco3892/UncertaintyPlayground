import unittest
import numpy as np
import torch
from uncertaintyplayground.utils.generate_data import generate_multi_modal_data
from uncertaintyplayground.trainers.mdn_trainer import MDNTrainer

class TestMDNTrainer(unittest.TestCase):
    """
    This test class provides unit tests for the MDNTrainer class.
    
    """

    def setUp(self):
        """
        Sets up the testing environment for each test method.
        """
        self.modes = [
            {'mean': -3.0, 'std_dev': 0.5, 'weight': 0.3},
            {'mean': 0.0, 'std_dev': 1.0, 'weight': 0.4},
            {'mean': 3.0, 'std_dev': 0.7, 'weight': 0.3}
        ]

        torch.manual_seed(1)
        np.random.seed(42)
        self.num_samples = 100
        self.X = np.random.rand(self.num_samples, 20)
        self.y = generate_multi_modal_data(self.num_samples, self.modes)

    def test_train(self):
        """
        Tests the train method of the MDNTrainer class.
        """
        trainer = MDNTrainer(self.X, self.y, num_epochs=3, lr=0.01, dense1_units=5, n_gaussians=3, dtype = torch.float32)
        trainer.train()

    def test_predict_with_uncertainty(self):
        """
        Tests the predict_with_uncertainty method of the MDNTrainer class.
        """
        trainer = MDNTrainer(self.X, self.y, num_epochs=2, lr=0.01, dense1_units=5, n_gaussians=3)
        trainer.train()

        # Generate a test instance
        test_instance = np.random.rand(1, 20)
        test_instance = test_instance.astype(np.float32)

        pi, mu, sigma, sample = trainer.predict_with_uncertainty(test_instance)
        # Add assertions based on the expected behavior of the predict_with_uncertainty function
        self.assertEqual(pi.shape, (1, 3))
        self.assertEqual(mu.shape, (1, 3))
        self.assertEqual(sigma.shape, (1, 3))
        self.assertIsInstance(sample, np.ndarray)

    # add other tests for other methods of MDNTrainer class

if __name__ == '__main__':
    unittest.main()
