# Import libraries
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error # for calculating R2 and MSE

torch.manual_seed(0) # for reproducibility
# Generate data with 50 predictors and 20000 observations
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
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points=train_x[:10, :], likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Initialize the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

# Assign sample weights to the loss function (optional)
# sample_weights = torch.ones(train_n) # change this to your desired weights
# sample_weights = torch.ones(train_n) # equal weights by default
# sample_weights = torch.rand(train_n) # random weights
sample_weights = 1 / train_y.abs() # inverse proportional to target magnitude

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
mll.register_forward_pre_hook(lambda mll, input: (input[0], input[1] * sample_weights))

# Train the model using Adam optimizer
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
training_iter = 50
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()
    # Calculate and print R2 and MSE metrics
    pred_y = output.mean.detach().numpy()
    true_y = train_y.numpy()
    r2 = r2_score(true_y, pred_y)
    mse = mean_squared_error(true_y, pred_y)
    print('R2: %.3f - MSE: %.3f' % (r2, mse))


# Test the model on new data and plot predictions
model.eval()
likelihood.eval()

# Compute mean absolute error on test set
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_pred = likelihood(model(test_x))
    mae = torch.mean(torch.abs(test_pred.mean - test_y))
    print('MAE: %.3f' % mae.item())
    # lower, upper = test_pred.confidence_region()
    # plt.figure(figsize=(12, 6))
    # plt.plot(test_y.numpy(), label='True values')
    # plt.plot(test_pred.mean.numpy(), label='Predicted values')
    # plt.fill_between(torch.arange(test_y.size(0)), lower.numpy(), upper.numpy(), alpha=0.5, label='Confidence interval')
    # plt.legend()
    # plt.show()