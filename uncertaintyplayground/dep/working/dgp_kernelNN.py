#-------------
# load the data
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = data['data']
y = data['target']

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#-------------
# define and run the model
import torch
import torch.nn as nn
import gpytorch
import pyro

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Define the input dimension
input_dim = X_train.shape[1]

# Number of inducing points
num_inducing = 100

# Define the neural network
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return torch.relu(self.layer2(x))

feature_extractor = FeatureExtractor(input_dim)

# Define the GP layer
class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_inducing=100):
        inducing_points = torch.randn(num_inducing, 64)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Define the DGP model
class DGP(gpytorch.Module):
    def __init__(self, feature_extractor, gp_layer):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = gp_layer

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.gp_layer(features)

dgp_model = DGP(feature_extractor, GPModel(num_inducing))

# Define the likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()

from sklearn.metrics import mean_squared_error, r2_score

# Training the model
optimizer = torch.optim.Adam(dgp_model.parameters(), lr=0.01)  # Change learning rate to 0
num_epochs = 200
dgp_model.train()
likelihood.train()

# Define the marginal log likelihood
mll = gpytorch.mlls.VariationalELBO(likelihood, dgp_model.gp_layer, num_data=len(X_train))

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = dgp_model(X_train_tensor)
    loss = -mll(output, y_train_tensor).sum()
    loss.backward()
    optimizer.step()

    # Print the epoch and loss during training
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

# Make predictions
dgp_model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    output = likelihood(dgp_model(X_test_tensor))
    y_pred_mean_dgp = output.mean
    y_pred_std_dgp = output.stddev
    y_pred_mean = output.mean

# Invert the scaling of predictions and standard deviations
y_pred_mean_dgp = scaler_y.inverse_transform(y_pred_mean_dgp.numpy().reshape(-1, 1))
y_pred_std_dgp = scaler_y.inverse_transform(y_pred_std_dgp.numpy().reshape(-1, 1))

# Invert the scaling of predictions
y_pred_mean = scaler_y.inverse_transform(y_pred_mean.numpy().reshape(-1, 1))

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred_mean)
r2 = r2_score(y_test, y_pred_mean)

# compare with a simple regression
from sklearn.linear_model import LinearRegression

# Train a linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions using the linear regression model
y_pred_lr = lr_model.predict(X_test)

# Calculate performance metrics for the linear regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Print performance metrics for the DGP model
print("DGP Model:")
print(f'Mean Squared Error (MSE): {mse}')
print(f'R2 Score: {r2}')

# Print performance metrics for the linear regression model
print("\nLinear Regression Model:")
print(f'Mean Squared Error (MSE): {mse_lr}')
print(f'R2 Score: {r2_lr}')

# make a plot fo the predictions
import matplotlib.pyplot as plt

def save_predictions_plot(y_true, y_pred, y_pred_std, filename, title='Predictions'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True values', linestyle='-', marker='o')
    plt.plot(y_pred, label='Predicted values', linestyle='-', marker='x')
    
    if y_pred_std is not None:
        # Calculate the 95% confidence intervals
        lower_bound = (y_pred - 1.96 * y_pred_std).reshape(-1)
        upper_bound = (y_pred + 1.96 * y_pred_std).reshape(-1)
        plt.fill_between(range(len(y_pred)), lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')
    
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Target Value')
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Save the plot for the DGP model
save_predictions_plot(y_test, y_pred_mean_dgp, y_pred_std_dgp, filename='dgp_model_predictions.png', title='DGP Model Predictions')

# Save the plot for the linear regression model
save_predictions_plot(y_test, y_pred_lr, None, filename='linear_regression_model_predictions.png', title='Linear Regression Model Predictions')
