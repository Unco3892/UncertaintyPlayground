# Import libraries
import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# Generate data with 50 predictors and 20000 observations
torch.manual_seed(0) # for reproducibility
n = 20000 # number of observations
d = 50 # number of predictors
X = torch.randn(n, d) # random input features
w = torch.randn(d) # random weights
y = X @ w + 0.1 * torch.randn(n) # linear output with noise

# Split data into train and test sets
train_n = int(0.8 * n) # 80% for training
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()
test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

# Define the GP model with inducing points
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points=torch.randn(10, d), likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Initialize the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

# Use a GPU if available
if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Define a function to assign sample weights to the loss function
def weighted_loss(output, target, weights):
    # output: MultivariateNormal distribution from model.forward()
    # target: tensor of target values
    # weights: tensor of sample weights
    # returns: a scalar tensor representing the weighted negative log likelihood

    # Get the mean and covariance matrix from output
    mean = output.mean
    covar = output.lazy_covariance_matrix

    # Compute the log determinant of covar
    log_det = covar.logdet()

    # Compute the Mahalanobis distance between target and mean
    diff = target - mean
    mahal = diff.unsqueeze(-2).matmul(covar.inv_matmul(diff.unsqueeze(-1))).squeeze(-1).squeeze(-1)

    # Compute the weighted negative log likelihood
    nll = -0.5 * (weights * (-log_det - mahal - math.log(2 * math.pi))).sum()

    return -nll

# Train the model with sample weights (here we use uniform weights for simplicity)
def train():
    model.train()
    likelihood.train()
    optimizer.zero_grad()

    # Generate sample weights (here we use uniform weights for simplicity)
    weights = torch.ones(train_n) / train_n

    # Get output from model
    output = model(train_x)

    # Compute weighted loss
    loss = weighted_loss(output, train_y, weights)

    # Backpropagate gradients
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))

    # Optimize model parameters
    optimizer.step()

# Run training for a number of iterations
training_iter = 5
for i in range(training_iter):
    print(i)
    train()

# # Set the model and likelihood to evaluation mode
# model.eval()
# likelihood.eval()

# # Make predictions on the test set
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(model(test_x))

# # Compute the root mean squared error (RMSE) on the test set
# rmse = torch.sqrt(torch.mean((observed_pred.mean - test_y) ** 2))
# print('RMSE on test set: %.3f' % rmse.item())

# # Plot the predictions vs the true values
# plt.figure(figsize=(12, 8))
# plt.scatter(test_y.cpu().numpy(), observed_pred.mean.cpu().numpy(), s=10)
# plt.plot([-3, 3], [-3, 3], 'k--')
# plt.xlabel('True values')
# plt.ylabel('Predicted values')
# plt.title('Sparse Gaussian Process Regression')
# plt.show()
    