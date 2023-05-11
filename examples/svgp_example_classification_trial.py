# Add some docstring after to all the classes
import torch
import gpytorch
import numpy as np
from sklearn.metrics import accuracy_score
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

    def __init__(self, inducing_points, num_classes, dtype=torch.float32):
        """
        Initialize the SVGP model.

        Args:
            inducing_points (torch.Tensor): Tensor containing the inducing points.
            num_classes (int): Number of classes in the classification problem.
            dtype (torch.dtype, optional): Data type for the model parameters. Defaults to torch.float32.
        """
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

        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward pass for the SVGP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            gpytorch.distributions.MultitaskMultivariateNormal: Multitask multivariate normal distribution with the given mean and covariance.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
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
        """
        Initialize the EarlyStopping object.

        Args:
            patience (int, optional): Number of consecutive epochs with no improvement after which training will be stopped. Defaults to 10.
            compare_fn (callable, optional): Function to compare two values of the validation metric to determine if one is better than the other. Defaults to lambda x, y: x < y.
        """
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
        self.best_val_accuracy = 0.0

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
            self.y = torch.tensor(self.y, dtype=torch.long)

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
    def __init__ (self, *args, num_inducing_points=100, num_classes=2, **kwargs):
        super().__init__ (*args, **kwargs)
        self.num_inducing_points = num_inducing_points
        self.num_classes = num_classes
        # Initialize the model with inducing points
        inducing_points = self.X_train[:num_inducing_points, :]
        self.model = SVGP(inducing_points, num_classes=self.num_classes, dtype=self.dtype).to(
            self.device, dtype=self.dtype)
        self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features = 1, num_classes=self.num_classes).to(
            self.device, dtype=self.dtype)
        
    def train(self):
        torch.manual_seed(self.random_state)
        # Define the optimizer & loss function (dynamically)
        optimizer_fn = getattr(torch.optim, self.optimizer_fn_name)
        optimizer = optimizer_fn(self.model.parameters(), lr=self.lr)

        # Define the learning rate scheduler if use_scheduler is True
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5, epochs=self.num_epochs,
                                                        steps_per_epoch=len(self.train_loader))

        # Define the loss function
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=self.X_train.shape[0])

        # Early stopping parameters
        early_stopping = EarlyStopping(
            patience=self.patience, compare_fn=lambda x, y: x > y)

        # Initiate the model training mode
        self.model.train()
        self.likelihood.train()

        for i in range(self.num_epochs):
            for X_batch, y_batch, weights_batch in self.train_loader:
                X_batch, y_batch, weights_batch = X_batch.to(self.device, dtype=self.dtype), y_batch.to(
                    self.device, dtype=torch.long), weights_batch.to(self.device, dtype=self.dtype)  # Move tensors to the chosen device
                optimizer.zero_grad()

                output = self.model(X_batch)
                unweighted_loss = -mll(output, y_batch)

                # Apply sample weights
                weighted_loss = torch.mean(unweighted_loss * weights_batch)

                weighted_loss.backward()

                optimizer.step()

                # Call the scheduler after the optimizer step
                if self.use_scheduler:
                    scheduler.step()

            # Compute validation metrics (accuracy)
            self.model.eval()
            self.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                y_pred_val = self.model(self.X_val).argmax(dim=-1)
            acc_val = accuracy_score(self.y_val.cpu().numpy(), y_pred_val.cpu().numpy())

            self.model.train()
            self.likelihood.train()

            print(
                f"Epoch {i + 1}/{self.num_epochs}, Weighted Loss: {weighted_loss.item():.3f}, Val Accuracy: {acc_val:.3f}")

            should_stop = early_stopping(acc_val, self.model)

            if should_stop:
                print(f"Early stopping after {i + 1} epochs")
                break

        if early_stopping.best_model_state is not None:
            self.model.load_state_dict(early_stopping.best_model_state)
            self.model.eval()
            self.likelihood.eval()

    def predict(self, X):
        """
        Predicts the class labels given input tensor X.

        Args:
            X (tensor): Input tensor of shape (num_samples, num_features).

        Returns:
            numpy array: The predicted class labels.
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
            # Get the predictive mean and convert to class labels
            preds = self.model(X).argmax(dim=-1)
            labels = preds.cpu().numpy()

        return labels

# Example usage of the SparseGPTrainer class

from sklearn.datasets import make_classification
# Create a multi-class classification dataset

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, n_classes=4, random_state=42)
# Convert labels to int64

y = y.astype(np.int64)
# Set seeds for reproducibility

torch.manual_seed(42)
np.random.seed(42)
# Create an instance of the SparseGPTrainer class

trainer = SparseGPTrainer(X=X, y=y, num_inducing_points=200, num_classes=4, num_epochs=5000, batch_size=1000, lr=0.2, patience=3)
# Train the model

trainer.train()
# Make predictions

y_pred = trainer.predict(trainer.X_val.numpy())
# Compute accuracy for validation set

accuracy = accuracy_score(trainer.y_val.numpy(), y_pred)
print('Accuracy:', accuracy)
