import unittest
import torch
import gpytorch
from models.svgp_model import SVGP

class TestSVGP(unittest.TestCase):
    """
    Unit tests for SVGP class.
    """

    def setUp(self):
        """
        Test fixture setup method.
        """
        self.inducing_points = torch.rand((10, 2))
        self.dtype = torch.float32
        self.svgp = SVGP(self.inducing_points, self.dtype)

    def test_init(self):
        """
        Test case for SVGP initialization.
        """
        self.assertIsInstance(self.svgp, SVGP)
        self.assertEqual(self.svgp.mean_module.constant, torch.tensor(0.))
        self.assertEqual(self.svgp.covar_module.base_kernel.lengthscale, torch.tensor(1.))

    def test_forward(self):
        """
        Test case for forward method in SVGP.
        """
        x = torch.rand((5, 2))
        output = self.svgp.forward(x)
        self.assertIsInstance(output, gpytorch.distributions.MultivariateNormal)
        self.assertEqual(output.loc.shape, x.shape[0])
        self.assertEqual(output.covariance_matrix.shape, (x.shape[0], x.shape[0]))


if __name__ == "__main__":
    unittest.main()
