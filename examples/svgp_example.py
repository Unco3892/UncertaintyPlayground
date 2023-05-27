# Add some docstring after to all the classes
import torch
import gpytorch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os


class SVGP(gpytorch.models.ApproximateGP):
    """
    Stochastic Variational Gaussian Process (SVGP) Model.

    A scalable Gaussian Process (GP) model based on stochastic variational inference.
    Inherits from the gpytorch.models.ApproximateGP class.

    Attributes:
        mean_module (gpytorch.means.ConstantMean): Constant mean module.
        covar_module (gpytorch.kernels.ScaleKernel): Scaled RBF kernel.
    """

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
        """
        Forward pass for the SVGP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            gpytorch.distributions.MultivariateNormal: Multivariate normal distribution with the given mean and covariance.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class EarlyStopping:
    """
    EarlyStopping is a utility class for early stopping during model training.

    Stops the training process when a specified performance metric does not improve for a specified number of consecutive epochs.

    Attributes:
        patience (int): Number of consecutive epochs with no improvement after which training will be stopped.
        counter (int): Current counter of the number of consecutive epochs with no improvement.
        best_val_metric (float): Best value of the validation metric observed so far.
        best_model_state (dict): Model state dictionary corresponding to the best validation metric.
        compare_fn (callable): Function to compare two values of the validation metric to determine if one is better than the other.
    """
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
            self.best_model_state = {k: v.clone()
                                     for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True

        return False

class BaseTrainer:
    def __init__(self, X, y, sample_weights=None, test_size=0.2, random_state=42, num_epochs=50, batch_size=256, optimizer_fn_name="Adam", lr=0.01, use_scheduler=False, patience=10, dtype=torch.float32):
        self.X = X
        self.y = y
        self.sample_weights = sample_weights
        self.test_size = test_size
        self.random_state = random_state
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer_fn_name = optimizer_fn_name
        self.lr = lr
        self.patience = patience
        self.dtype = dtype
        self.use_scheduler = use_scheduler

        # setting for early stopping
        self.best_epoch = -1
        self.best_val_mse = np.inf

        # Choose device (GPU if available, otherwise CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Convert input tensors to the correct type
        self.prepare_inputs()

        # Split data into training and validation sets
        self.split_data()

        # Create DataLoader for training data
        self.prepare_dataloader()

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

    def split_data(self, test_size=0.2):
        if self.sample_weights is None:
            self.sample_weights = torch.ones(self.X.shape[0], dtype=self.dtype)

        self.X_train, self.X_val, self.y_train, self.y_val, self.sample_weights_train, self.sample_weights_val = train_test_split(
            self.X, self.y, self.sample_weights, test_size=test_size, random_state=self.random_state
        )

    def custom_lr_scheduler(self, epoch):
        if epoch < 3:
            return 1 - 0.1 * epoch / self.lr
        else:
            return 0.2 / self.lr

    def prepare_dataloader(self):
        # Use all available CPU cores or default to 1 if not detected
        num_workers = os.cpu_count()-1 or 1
        train_dataset = TensorDataset(
            self.X_train, self.y_train, self.sample_weights_train)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


class SparseGPTrainer(BaseTrainer):
    def __init__(self, *args, num_inducing_points=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_inducing_points = num_inducing_points
        
        # Initialize the model with inducing points
        inducing_points = self.X_train[:num_inducing_points, :]
        self.model = SVGP(inducing_points, dtype=self.dtype).to(
            self.device, dtype=self.dtype)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            dtype=self.dtype).to(self.device, dtype=self.dtype)
        
    def train(self):
        # set the seed
        torch.manual_seed(self.random_state)

        # define the optimizer & loss function (dynamically)
        optimizer_fn = getattr(torch.optim, self.optimizer_fn_name)
        optimizer = optimizer_fn(self.model.parameters(), lr=self.lr)
        
        # can use either one of the schuedlers
        # define the learning rate scheduler if use_scheduler is True
        # if self.use_scheduler:
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         optimizer, self.custom_lr_scheduler)
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 5, epochs=50,  steps_per_epoch=len(self.train_loader))

        # define the loss function
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=self.X_train.shape[0])

        # Early stopping parameters
        early_stopping = EarlyStopping(
            patience=self.patience, compare_fn=lambda x, y: x < y)

        # Initiate the model training mode
        self.model.train()
        self.likelihood.train()

        for i in range(self.num_epochs):
            for X_batch, y_batch, weights_batch in self.train_loader:
                X_batch, y_batch, weights_batch = X_batch.to(self.device, dtype=self.dtype), y_batch.to(
                    self.device, dtype=self.dtype), weights_batch.to(self.device, dtype=self.dtype)  # Move tensors to the chosen device
                optimizer.zero_grad()

                output = self.model(X_batch)
                unweighted_loss = -mll(output, y_batch)

                # Apply sample weights
                weighted_loss = torch.mean(unweighted_loss * weights_batch)

                weighted_loss.backward()

                optimizer.step()
                
                # the scheduler is called after the optimizer
                if self.use_scheduler:
                    scheduler.step()    
            
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
                f"Epoch {i + 1}/{self.num_epochs}, Weighted Loss: {weighted_loss.item():.3f}, Val MSE: {mse_val:.6f}, Val R2: {r2_val:.3f}")

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

        # Convert numpy array to PyTorch tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.dtype)

        # Check if X is a single instance and add an extra dimension if necessary
        if X.ndim == 1:
            X = torch.unsqueeze(X, 0)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get the predictive mean and variance
            preds = self.likelihood(self.model(X))
            mean = preds.mean.cpu().numpy()
            variance = preds.variance.cpu().numpy()

        return mean, variance

# Example usage of the SVGPR class

# Assuming that your data is numpy arrays
X = np.random.rand(1000, 20)
y = np.random.rand(1000)

# we also set the seeds for the models
torch.manual_seed(42)
np.random.seed(42)

# Create an instance of the SparseGPTrainer class
trainer = SparseGPTrainer(X=X, y=y, num_inducing_points=200, num_epochs=5000, batch_size=1000, lr=0.2, patience=3)

# Train the model
trainer.train()

# Make predictions with uncertainty
y_pred, y_var = trainer.predict_with_uncertainty(trainer.X_val.numpy())

# Compute R2 score for validation set
print('R2 Score:', r2_score(trainer.y_val, y_pred))

#------------------------------------------------------------
# Example of plotting the predictions with uncertainty
# import matplotlib.pyplot as plt

# # choose an instance from validation set

# # predict mean and variance for the instance
# y_pred, y_var = trainer.predict_with_uncertainty(x_instance)

# # calculate standard deviation for the prediction
# y_std = np.sqrt(y_var)

# # plot prediction and confidence interval
# print(x_instance)
# print(y_pred)
# plt.figure(figsize=(10, 5))
# plt.plot(x_instance, y_pred, 'ro', markersize=10, label='Prediction')
# plt.fill_between(x_instance, y_pred - y_std, y_pred + y_std, color='r', alpha=0.2, label='Confidence Interval')
# plt.legend()
# plt.show(

x_instance = trainer.X_val[0].numpy()  # First instance from validation set
y_instance = trainer.y_val[0].numpy()  # First instance from validation set


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def compare_distributions_svgpr(trainer, x_instance, y_actual=None, num_samples=10000, ax=None):
    """
    Compare the actual and predicted outcome distributions.

    Args:
        trainer (SVGPRTrainer): The trained SVGPRTrainer instance.
        x_instance (np.ndarray): The instance for which to predict the outcome distribution.
        y_actual (float or np.ndarray, optional): The actual outcome. If a single value, plot as a vertical line.
                                                  If an array or list, plot as a KDE. If None, don't plot actual outcome.
        num_samples (int, optional): The number of samples to generate from the predicted distribution.
                                     Default is 10000.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None, create a new figure.

    Returns:
        None
    """
    # Ensure x_instance is a 2D array
    if x_instance.ndim == 1:
        x_instance = np.expand_dims(x_instance, axis=0)

    # Get the predicted mean and standard deviation
    mu, sigma = trainer.predict_with_uncertainty(x_instance)

    # Generate samples from the predicted distribution
    predicted_samples = np.random.normal(mu, sigma, num_samples)

    # Plot KDE of predicted samples
    if ax is None:
        sns.kdeplot(predicted_samples, fill=True, color="r", label="Predicted distribution")
        plt.axvline(mu, color="r", linestyle="--", label="Predicted value")
    else:
        sns.kdeplot(predicted_samples, fill=True, color="r", label="Predicted distribution", ax=ax)
        ax.axvline(mu, color="r", linestyle="--", label="Predicted value")

    # Plot the actual value
    if y_actual is not None:
        if ax is None:
            plt.axvline(y_actual, color="b", linestyle="--", label="Actual value")
        else:
            ax.axvline(y_actual, color="b", linestyle="--", label="Actual value")
    
    # Set an option when doing multiple plots
    if ax is None:
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.show()


def plot_results_grid_svgpr(trainer, X_test, Y_test, indices, ncols=2, dtype=np.float32):
    """
    Plot a grid of comparison plots (minimum 2) for a set of test instances.

    Args:
        trainer (SVGPRTrainer): The trained SVGPRTrainer instance.
        X_test (np.ndarray): The test input data of shape (num_samples, num_features).
        Y_test (np.ndarray): The test target data of shape (num_samples,).
        indices (list): The indices of the instances to plot.
        ncols (int, optional): Number of columns in the grid. Default is 2.
        dtype (np.dtype, optional): Data type to use for plotting. Default is np.float32.

    Returns:
        None
    """
    num_instances = len(indices)
    nrows = (num_instances - 1) // ncols + 1

    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

    for i, ax in zip(indices, axes.flat):
        x_instance = X_test[i].astype(dtype)
        y_actual = Y_test[i].astype(dtype)
        compare_distributions_svgpr(trainer, x_instance, y_actual, ax=ax)
        ax.set_title(f"Test Instance: {i}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(axis='y', alpha=0.75)

    # Remove empty subplots
    if num_instances < nrows * ncols:
        for ax in axes.flat[num_instances:]:
            ax.remove()

    plt.tight_layout()
    plt.show()

# Assume you have a trained SVGPRTrainer, test set X_test and Y_test, and you want to plot for indices 0, 1, 2
compare_distributions_svgpr(trainer, x_instance, y_instance)

# Call the plot_results_grid_svgpr function to plot distributions for multiple instances
indices = [0, 1, 2, 3, 4]  # Indices of instances to plot
plot_results_grid_svgpr(trainer, trainer.X_val.numpy(), trainer.y_val.numpy(), indices)