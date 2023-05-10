# Import modules
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error # for evaluation metrics

# Generate data where there are 50 predictors and 20,000 observations
torch.manual_seed(0) # for reproducibility
n = 20000 # number of observations
p = 50 # number of predictors
X = torch.randn(n, p) # random features
w = torch.randn(p) # random weights
y = X @ w + 0.1 * torch.randn(n) # linear model with noise

# Split data into train and test sets
train_n = int(0.8 * n) # 80% for training
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()
test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

# Assign sample weights (optional)
# sample_weight = torch.ones(train_n) # equal weights by default
sample_weight = torch.rand(train_n) # random weights
# sample_weight = 1 / train_y.abs() # inverse proportional to target magnitude

# Define the GP regression model with inducing point kernel
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

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Loss function is the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

def train():
    num_iter = 50 # number of iterations
    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y, sample_weight) # use sample weights here
        loss.backward()
        
        # Print loss, R2 and MSE during training
        print('Iter %d/%d - Loss: %.3f' % (i + 1, num_iter, loss.item()))
        pred_y = output.mean.detach().numpy()
        r2 = r2_score(train_y.numpy(), pred_y)
        mse = mean_squared_error(train_y.numpy(), pred_y)
        print('R2: %.3f - MSE: %.3f' % (r2, mse))
        
        optimizer.step()

# Train the model
# %time train()
train()

# Make predictions with the model
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))

# Compute mean absolute error on test set
mae = torch.mean(torch.abs(observed_pred.mean - test_y))
print('MAE: %.3f' % mae.item())
plt.clf()

# Plot predictions and confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(test_y.numpy(), label='True test y')
plt.plot(observed_pred.mean.numpy(), label='Predicted test y')
plt.fill_between(torch.arange(test_y.size(0)), observed_pred.confidence_region()[0].numpy(), observed_pred.confidence_region()[1].numpy(), alpha=0.5, label='Confidence interval')
plt.legend()
plt.show()