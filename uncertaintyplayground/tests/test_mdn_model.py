import unittest
import torch
from uncertaintyplayground.models.mdn_model import MDN

class TestMDN(unittest.TestCase):
    """Tests for the class MDN"""

    def setUp(self):
        """Set up a test fixture with input_dim = 20, dense1_units  = 10, n_gaussians = 3"""
        self.input_dim = 20
        self.dense1_units = 10
        self.n_gaussians = 3
        self.mdn = MDN(input_dim=self.input_dim, n_gaussians=self.n_gaussians, dense1_units=self.dense1_units)

    def test_init(self):
        """Test that the MDN is initialized properly"""
        self.assertEqual(self.mdn.z_h[0].in_features, self.input_dim)
        self.assertEqual(self.mdn.z_h[0].out_features, self.dense1_units)
        self.assertEqual(self.mdn.z_pi.in_features, self.dense1_units)
        self.assertEqual(self.mdn.z_pi.out_features, self.n_gaussians)

    def test_forward(self):
        """Test the forward function with a tensor of shape (1, 20)"""
        x = torch.rand((1, self.input_dim))
        pi, mu, sigma = self.mdn.forward(x)
        self.assertEqual(pi.shape, (1, self.n_gaussians))
        self.assertEqual(mu.shape, (1, self.n_gaussians))
        self.assertEqual(sigma.shape, (1, self.n_gaussians))

    def test_sample(self):
        """Test the sample function with a tensor of shape (1, 20)"""
        x = torch.rand((1, self.input_dim))
        sample = self.mdn.sample(x)
        self.assertEqual(sample.shape, (1,))

if __name__ == "__main__":
    unittest.main()
