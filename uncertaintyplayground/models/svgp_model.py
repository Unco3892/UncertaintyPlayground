import torch
import gpytorch

class SVGP(gpytorch.models.ApproximateGP):
    """
    Stochastic Variational Gaussian Process (SVGP) Regression Model.

    A scalable Gaussian Process (GP) model based on stochastic variational inference.
    Inherits from the gpytorch.models.ApproximateGP class.

    Args:
        inducing_points (torch.Tensor): Inducing points tensor.
        dtype (torch.dtype, optional): Data type of the model. Defaults to torch.float32.
        device (torch.device): Device can be specified to the desired `cpu` or `cuda` for GPU
    
    Attributes:
        mean_module (gpytorch.means.ConstantMean): Constant mean module.
        covar_module (gpytorch.kernels.ScaleKernel): Scaled RBF kernel.
    """

    def __init__(self, inducing_points, dtype=torch.float32, device = None):
        self.device = device
        self.inducing_points = inducing_points.to(device = self.device)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            self.inducing_points.size(0), dtype=dtype
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            self.inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        ).to(dtype = dtype)

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(dtype=dtype)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        """
        Forward pass for the SVGPRegressionModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            gpytorch.distributions.MultivariateNormal: Multivariate normal distribution with the given mean and covariance.
        """
        x = x.to(device = self.device)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
