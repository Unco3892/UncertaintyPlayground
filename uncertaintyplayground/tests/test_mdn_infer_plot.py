import unittest
import numpy as np
import torch
from trainers.mdn_trainer import MDNTrainer
from utils.generate_data import generate_multi_modal_data
from utils.mdn_infer_plot import compare_distributions, plot_results_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DisablePlotDisplay:
    """
    Context manager to disable the display of matplotlib plots.
    """

    def __enter__(self):
        plt.ioff()  # Turn off interactive mode
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.close()  # Close the plot


class TestDistributionComparison(unittest.TestCase):
    """
    This test class provides unit tests for the compare_distributions and plot_results_grid functions.
    
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
        self.num_samples = 1000
        self.X_test = np.random.rand(self.num_samples, 20)
        self.Y_test = generate_multi_modal_data(self.num_samples, self.modes)

        # Initialize and train the MDN trainer
        self.trainer = MDNTrainer(self.X_test, self.Y_test, num_epochs=100, lr=0.01, n_hidden=20, n_gaussians=3)
        self.trainer.train()

    def test_compare_distributions(self):
        """
        Tests the compare_distributions function.
        """
        index_instance = 900
        test_instance = self.X_test[index_instance, :]
        test_instance = test_instance.astype(np.float32)

        # Disable the display of plots within the test
        with DisablePlotDisplay():
            # Call the function without asserting the return value
            compare_distributions(self.trainer, test_instance, y_actual=self.Y_test[index_instance])

    def test_plot_results_grid(self):
        """
        Tests the plot_results_grid function.
        """
        indices = [900, 100]  # Example indices
        
        # Disable the display of plots within the test
        with DisablePlotDisplay():
            # Call the function without showing the plot
            plot_results_grid(self.trainer, self.X_test, self.Y_test, indices)


if __name__ == '__main__':
    unittest.main()
