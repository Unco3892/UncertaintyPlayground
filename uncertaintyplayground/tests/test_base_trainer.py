import unittest
import torch
from uncertaintyplayground.trainers.base_trainer import BaseTrainer

class TestBaseTrainer(unittest.TestCase):
    """
    Unit test suite for the BaseTrainer class.

    This class contains a series of methods to test the functionalities of BaseTrainer, including preparing the data loader.

    Attributes:
        X_train (torch.Tensor): A tensor of feature vectors for the training data.
        y_train (torch.Tensor): A tensor of target values for the training data.
        sample_weights_train (torch.Tensor): A tensor of sample weights for the training data.
        trainer (BaseTrainer): The BaseTrainer instance to test.
    """

    def setUp(self):
        """
        Setup function that runs before each test method.

        This method generates random data for the tests and initializes an instance of BaseTrainer.
        """
        self.X = torch.randn(100, 20)
        self.y = torch.randn(100)
        self.batch_size = 10
        self.test_size = 0.2
        self.sample_weights = torch.randn(100)
        self.trainer = BaseTrainer(self.X, self.y, self.sample_weights, batch_size=self.batch_size, test_size= self.test_size)

    def test_prepare_dataloader(self):
        """
        Tests the prepare_dataloader method of the BaseTrainer class.

        This test prepares the data loader and checks that it has the correct length and batch size.
        """
        self.trainer.prepare_dataloader()
        self.assertEqual(len(self.trainer.train_loader) * self.batch_size, len(self.trainer.X_train))

        for batch in self.trainer.train_loader:
            self.assertEqual(batch[0].shape, (self.batch_size, self.X.shape[1]))
            self.assertEqual(batch[1].shape, (self.batch_size,))
            self.assertEqual(batch[2].shape, (self.batch_size,))

if __name__ == "__main__":
    unittest.main()