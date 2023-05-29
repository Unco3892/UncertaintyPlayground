import torch
import numpy as np
from uncertaintyplayground.trainers.base_trainer import BaseTrainer
from uncertaintyplayground.models.mdn_model import MDN, mdn_loss
from uncertaintyplayground.utils.early_stopping import EarlyStopping

class MDNTrainer(BaseTrainer):
    """
    Trainer for the Mixed Density Network (MDN) model.

    This class handles the training process for the MDN model.

    Args:
        dense1_units (int): Number of hidden units in first layer of the neural network.
        n_gaussians (int): Number of Gaussian components in the mixture.
        **kwargs: Additional arguments passed to the BaseTrainer.

    Attributes:
        n_gaussians (int): Number of Gaussian components in the mixture.
        model (MDN): The MDN model.
        optimizer (torch.optim.Optimizer): The optimizer for model training.
    """

    def __init__(self, *args, dense1_units=20, n_gaussians=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gaussians = n_gaussians

        self.model = MDN(input_dim=self.X.shape[1], n_gaussians=self.n_gaussians, dense1_units = dense1_units).to(self.device)
        if self.dtype == torch.float64:
            self.model = self.model.double()  # Convert model parameters to float64
        optimizer_fn = getattr(torch.optim, self.optimizer_fn_name)
        self.optimizer = optimizer_fn(self.model.parameters(), lr=self.lr)

    def train(self):
        """
        Train the MDN model.
        """
        self.model.train()
        early_stopping = EarlyStopping(patience=self.patience, compare_fn=lambda x, y: x < y)

        for epoch in range(self.num_epochs):
            for X_batch, y_batch, weights_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()

                pi, mu, sigma = self.model(X_batch)
                loss = mdn_loss(y_batch, mu, sigma, pi)

                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                pi, mu, sigma = self.model(self.X_val.to(self.device))
                val_loss = mdn_loss(self.y_val.to(self.device), mu, sigma, pi)

            self.model.train()

            print(
                f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {loss.item():.3f}, "
                f"Validation Loss: {val_loss.item():.3f}"
            )

            should_stop = early_stopping(val_loss.item(), self.model)

            if should_stop:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        if early_stopping.best_model_state is not None:
            self.model.load_state_dict(early_stopping.best_model_state)
            self.model.eval()

    def predict_with_uncertainty(self, X):
        """
        Predict the output distribution given input data.

        Args:
            X (np.ndarray or torch.Tensor): Input data of shape (num_samples, num_features).

        Returns:
            tuple: A tuple containing the predicted mixture weights, means, standard deviations, and samples.
        """
        self.model.eval()

        # Convert numpy array to PyTorch tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.device)

        # Check if X is a single instance and add an extra dimension if necessary
        if X.ndim == 1:
            X = torch.unsqueeze(X, 0)

        with torch.no_grad():
            pi, mu, sigma = self.model(X)
            sample = self.model.sample(X, num_samples=1000)

        return pi.cpu().numpy(), mu.cpu().numpy(), sigma.cpu().numpy(), sample.cpu().numpy()

