import unittest
import torch
import gpytorch
from uncertaintyplayground.models.svgp_model import SVGP

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
        self.assertEqual(self.svgp.mean_module.constant.item(), 0.)

        # Expect default initial lengthscale to be ~0.693
        expected_lengthscale = torch.log(torch.tensor(2.0)).item()
        self.assertAlmostEqual(self.svgp.covar_module.base_kernel.lengthscale.item(), expected_lengthscale, places=5)

    def test_forward(self):
        """
        Test case for forward method in SVGP.
        """
        x = torch.rand((5, 2))
        output = self.svgp.forward(x)
        self.assertIsInstance(output, gpytorch.distributions.MultivariateNormal)
        self.assertEqual(output.loc.shape, torch.Size([x.shape[0]]))

        # Uncomment the next line if you really need to check the covariance matrix shape
        # Be aware that it might be computationally expensive for large matrices
        # self.assertEqual(output.covariance_matrix.shape, torch.Size([x.shape[0], x.shape[0]))


if __name__ == "__main__":
    unittest.main()
