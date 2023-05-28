import unittest
import torch
from uncertaintyplayground.models.mdn_model import MDN

class TestMDN(unittest.TestCase):
    """Tests for the class MDN"""

    def setUp(self):
        """Set up a test fixture with n_hidden = 20, n_gaussians = 3"""
        self.mdn = MDN(n_hidden=20, n_gaussians=3)

    def test_init(self):
        """Test that the MDN is initialized properly"""
        self.assertEqual(self.mdn.z_h[0].in_features, 20)
        self.assertEqual(self.mdn.z_h[0].out_features, 20)
        self.assertEqual(self.mdn.z_pi.in_features, 20)
        self.assertEqual(self.mdn.z_pi.out_features, 3)

    def test_forward(self):
        """Test the forward function with a tensor of shape (1, 20)"""
        x = torch.rand((1, 20))
        pi, mu, sigma = self.mdn.forward(x)
        self.assertEqual(pi.shape, (1, 3))
        self.assertEqual(mu.shape, (1, 3))
        self.assertEqual(sigma.shape, (1, 3))

    def test_sample(self):
        """Test the sample function with a tensor of shape (1, 20)"""
        x = torch.rand((1, 20))
        sample = self.mdn.sample(x)
        self.assertEqual(sample.shape, (1,))

if __name__ == "__main__":
    unittest.main()
