
# class TestCompareDistributions(unittest.TestCase):
#     """
#     Unit test class for compare_distributions function.

#     This class contains unit tests for the compare_distributions function. It tests whether
#     the function can handle both single and multiple instances.
#     """
#     @classmethod
#     def setUpClass(cls):
#         """
#         Set up data and MDNTrainer instance for the tests.

#         This method generates some training data, initializes an MDNTrainer, and trains it.
#         """
#         torch.manual_seed(1)
#         np.random.seed(42)
#         num_samples = 1000
#         cls.modes = [
#             {'mean': -3.0, 'std_dev': 0.5, 'weight': 0.3},
#             {'mean': 0.0, 'std_dev': 1.0, 'weight': 0.4},
#             {'mean': 3.0, 'std_dev': 0.7, 'weight': 0.3}
#         ]
#         X = np.random.rand(num_samples, 20)
#         y = generate_multi_modal_data(num_samples, cls.modes)
#         cls.trainer = MDNTrainer(X, y, num_epochs=100, lr=0.01, n_hidden=20, n_gaussians=3)
#         cls.trainer.train()
#         cls.test_instance = np.random.rand(20).astype(np.float32)

