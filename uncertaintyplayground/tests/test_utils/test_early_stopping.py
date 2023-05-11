import unittest
import torch

class TestEarlyStopping(unittest.TestCase):
    def setUp(self):
        self.early_stopping = EarlyStopping()

    def test_init(self):
        self.assertIsInstance(self.early_stopping, EarlyStopping)
        self.assertEqual(self.early_stopping.counter, 0)
        self.assertEqual(self.early_stopping.best_val_metric, np.inf)

    def test_call(self):
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
