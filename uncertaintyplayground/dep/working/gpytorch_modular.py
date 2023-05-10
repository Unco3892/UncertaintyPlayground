# Add some docstring after to all the classes
import torch
import gpytorch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class SVGPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, dtype=torch.float32):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), dtype=dtype
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        ).to(dtype)
        
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(dtype=dtype)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class EarlyStopping:
    def __init__(self, patience=10, compare_fn=lambda x, y: x < y):
        self.patience = patience
        self.counter = 0
        self.best_val_metric = np.inf
        self.best_model_state = None
        self.compare_fn = compare_fn

    def __call__(self, val_metric, model):
        if self.compare_fn(val_metric, self.best_val_metric):
            self.best_val_metric = val_metric
            self.counter = 0
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True

        return False


class SparseGPTrainer:
    def __init__(self, X, y, sample_weights=None, num_inducing_points=100, test_size=0.2, random_state=42, num_epochs=50, batch_size=256, lr=0.01, patience=50,dtype=torch.float32):
        self.X = X
        self.y = y
        self.sample_weights = sample_weights
        self.num_inducing_points = num_inducing_points
        self.test_size = test_size
        self.random_state = random_state
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.dtype = dtype
        
        # setting for early stopping
        self.best_epoch = -1
        self.best_val_mse = np.inf
        
        # Convert input tensors to the correct type
        self.prepare_inputs()
        
        # Split data into training and validation sets
        self.split_data()

        # Initialize the model with inducing points
        inducing_points = self.X_train[:num_inducing_points, :]
        self.model = SVGPRegressionModel(inducing_points, dtype=self.dtype)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(dtype=self.dtype)
    
    def prepare_inputs(self):
        # Convert X to a tensor if it's a numpy array
        if isinstance(self.X, np.ndarray):
            self.X = torch.from_numpy(self.X).to(self.dtype)

        # Convert y to a tensor if it's a list or numpy array
        if isinstance(self.y, (list, np.ndarray)):
            self.y = torch.tensor(self.y, dtype=self.dtype)

        # Check if sample_weights is a tensor, numpy array, or list
        if self.sample_weights is not None:
            if isinstance(self.sample_weights, (np.ndarray, list)):
                self.sample_weights = torch.tensor(
                    self.sample_weights, dtype=self.dtype)
                
    def split_data(self, test_size=0.2, random_state=42):
        if self.sample_weights is None:
            self.sample_weights = torch.ones(self.X.shape[0],dtype = self.dtype)

        self.X_train, self.X_val, self.y_train, self.y_val, self.sample_weights_train, self.sample_weights_val = train_test_split(
            self.X, self.y, self.sample_weights, test_size=test_size, random_state=random_state
        )
        

    def train(self):
        # set the seed
        torch.manual_seed(self.random_state)

        # define the optimizer & loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=self.X_train.shape[0])

        self.model.train()
        self.likelihood.train()

        # Early stopping parameters
        early_stopping = EarlyStopping(patience=self.patience, compare_fn=lambda x, y: x < y)

        for i in range(self.num_epochs):
            batch_indices = torch.randperm(self.X_train.shape[0])[
                :self.batch_size]
            X_batch = self.X_train[batch_indices]
            y_batch = self.y_train[batch_indices]

            if self.sample_weights is not None:
                weights_batch = self.sample_weights_train[batch_indices]
            else:
                weights_batch = torch.ones(self.batch_size)

            optimizer.zero_grad()

            output = self.model(X_batch)
            unweighted_loss = -mll(output, y_batch)

            # Apply sample weights
            weighted_loss = torch.mean(unweighted_loss * weights_batch)

            weighted_loss.backward()

            optimizer.step()

            # Compute validation metrics (MSE and R2)
            self.model.eval()
            self.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                y_pred_val = self.likelihood(self.model(self.X_val)).mean

            mse_val = mean_squared_error(
                self.y_val.detach().numpy(), y_pred_val.detach().numpy())
            r2_val = r2_score(self.y_val.detach().numpy(),
                            y_pred_val.detach().numpy())

            self.model.train()
            self.likelihood.train()

            print(
                f"Epoch {i + 1}/{self.num_epochs}, Weighted Loss: {weighted_loss.item():.3f}, Val MSE: {mse_val:.3f}, Val R2: {r2_val:.3f}")

            should_stop = early_stopping(mse_val, self.model)

            if should_stop:
                print(f"Early stopping after {i + 1} epochs")
                break

        if early_stopping.best_model_state is not None:
            self.model.load_state_dict(early_stopping.best_model_state)
            self.model.eval()
            self.likelihood.eval()

    def predict_with_uncertainty(self, X):
        """
        Predicts the mean and variance of the output distribution given input tensor X.

        Args:
            X (tensor): Input tensor of shape (num_samples, num_features).

        Returns:
            tuple: A tuple of the mean and variance of the output distribution, both of shape (num_samples,).
        """
        self.model.eval()
        self.likelihood.eval()
        
        # Check if X is a single instance and add an extra dimension if necessary
        if X.ndim == 1:
            X = torch.unsqueeze(X, 0)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get the predictive mean and variance
            preds = self.likelihood(self.model(X))
            mean = preds.mean.cpu().numpy()
            variance = preds.variance.cpu().numpy()

        return mean, variance


# we also set the seeds for the models
# torch.manual_seed(42)
# np.random.seed(42)
# SparseGPTrainer(X, y).train()

# x = torch.from_numpy(r.X).float()
# y = torch.tensor(r.y, dtype=torch.float32)
# a = SparseGPTrainer(x, y, num_epochs=5000, batch_size=800, lr=0.5)
mod = SparseGPTrainer(r.X, r.y, num_epochs=5000, batch_size=1000, lr=0.2, dtype=torch.float32)
mod.train()

# testing with sample weights
# sample_weights = torch.ones(r.X.shape[0]) # change this to your desired weights
# sample_weights = torch.ones(r.X.shape[0]) # equal weights by default
# sample_weights = torch.rand(r.X.shape[0]) # random weights
# sample_weights = 1 / r.X[0].abs() # inverse proportional to target magnitude
# mod = SparseGPTrainer(r.X, r.y, sample_weights = sample_weights, num_epochs=5000, batch_size=800, lr=0.5, dtype=torch.float32)
# mod.train()

# Add an extra dimension to the input tensor
# X = torch.unsqueeze(mod.X_val[1], 0)

# Make the prediction (group or single)
y_pred, y_var = mod.predict_with_uncertainty(mod.X_val)
# y_pred, y_var = mod.predict_with_uncertainty(mod.X_val[1])
r2_score(a.y_val, y_pred)

