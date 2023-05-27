import unittest
import torch
import numpy as np
from trainers.svgp_trainer import SparseGPTrainer
from utils.generate_data import generate_multi_modal_data
from uncertaintyplayground.predplot.svgp_predplot import compare_distributions_svgpr
from uncertaintyplayground.predplot.grid_predplot import plot_results_grid, DisablePlotDisplay
import matplotlib
matplotlib.use('Agg')

class TestSVGPRPlots(unittest.TestCase):
    """
    This test class provides unit tests for the compare_distributions_svgpr and plot_results_grid functions for SVGPR model.
    """

    def setUp(self):
        """
        Sets up the testing environment for each test method.
        """
        # Assuming data is generated in a similar way as in the MDN case
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

        self.svgpr_trainer = SparseGPTrainer(self.X_test, self.Y_test, num_epochs=5, lr=0.01)
        self.svgpr_trainer.train()

    def test_compare_distributions_svgpr(self):
        """
        Tests the compare_distributions function for the SVGPR model.
        """
        index_instance = 900
        test_instance = self.X_test[index_instance, :]
        test_instance = test_instance.astype(np.float32)

        with DisablePlotDisplay():
            compare_distributions_svgpr(self.svgpr_trainer, test_instance, y_actual=self.Y_test[index_instance])

    def test_plot_results_grid_svgpr(self):
        """
        Tests the plot_results_grid function with SVGPR model.
        """
        indices = [900, 100]  # Example indices

        with DisablePlotDisplay():
            plot_results_grid(self.svgpr_trainer, compare_distributions_svgpr, self.X_test, self.Y_test, indices)

if __name__ == '__main__':
    unittest.main()
