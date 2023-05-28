import unittest
import torch
import numpy as np
from uncertaintyplayground.utils.early_stopping import EarlyStopping

class TestEarlyStopping(unittest.TestCase):
    """
    Unit tests for EarlyStopping class.
    """

    def setUp(self):
        """
        Test fixture setup method.
        """
        self.early_stopping = EarlyStopping()

    def test_init(self):
        """
        Test case for EarlyStopping initialization.
        """
        self.assertIsInstance(self.early_stopping, EarlyStopping)
        self.assertEqual(self.early_stopping.counter, 0)
        self.assertEqual(self.early_stopping.best_val_metric, np.inf)

    def test_call(self):
        """
        Test case for call method in EarlyStopping.
        """
        model = torch.nn.Linear(1, 1)  # simple model for testing
        val_metric = 10
        self.early_stopping(val_metric, model)
        self.assertEqual(self.early_stopping.best_val_metric, 10)
        self.assertEqual(self.early_stopping.counter, 0)

        val_metric = 20
        self.early_stopping(val_metric, model)
        self.assertEqual(self.early_stopping.best_val_metric, 10)
        self.assertEqual(self.early_stopping.counter, 1)

if __name__ == "__main__":
    unittest.main()
