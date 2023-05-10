import torch
import gpytorch
import numpy as np

# Generate synthetic data with 50 predictors and 20,000 observations
num_inputs = 50
num_data = 20000
num_inducing_points = 100

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate input features
X = torch.randn(num_data, num_inputs)

# Generate sample weights
sample_weights = torch.randn(num_data)

# Define a simple ground truth function
def ground_truth(X):
    return torch.sin(X[:, 0]) + torch.cos(X[:, 1])

# Generate noisy observations
y = ground_truth(X) + 0.1 * torch.randn(num_data)

# Model definition
class SVGPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Initialize the model with inducing points
inducing_points = X[:num_inducing_points, :]
model = SVGPRegressionModel(inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()


# Training loop
num_epochs = 50
batch_size = 256
lr = 0.01
use_sample_weights = True

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)

model.train()
likelihood.train()

for i in range(num_epochs):
    batch_indices = torch.randperm(num_data)[:batch_size]
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]
    
    if use_sample_weights:
        weights_batch = sample_weights[batch_indices]
    else:
        weights_batch = torch.ones(batch_size)
    
    optimizer.zero_grad()
    
    output = model(X_batch)
    unweighted_loss = -mll(output, y_batch)
    
    # Apply sample weights
    weighted_loss = torch.mean(unweighted_loss * weights_batch)
    
    weighted_loss.backward()
    
    optimizer.step()

    print(f"Epoch {i + 1}/{num_epochs}, Weighted Loss: {weighted_loss.item()}")

# Model evaluation
# model.eval()
# likelihood.eval()

# Make predictions on a new set of data points
# X_test = torch
