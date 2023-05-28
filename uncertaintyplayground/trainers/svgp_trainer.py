import torch
import gpytorch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from uncertaintyplayground.trainers.base_trainer import BaseTrainer
from uncertaintyplayground.models.svgp_model import SVGP
from uncertaintyplayground.utils.early_stopping import EarlyStopping

class SparseGPTrainer(BaseTrainer):
    """
    Trains an SVGP model using specified parameters and early stopping.
    
    Attributes:
        num_inducing_points (int): Number of inducing points for the SVGP.
        model (SVGP): The Stochastic Variational Gaussian Process model.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): The likelihood of the model.
    
    Args:
        X (array-like): The input features.
        y (array-like): The target outputs.
        num_inducing_points (int): Number of inducing points to use in the SVGP model.
        sample_weights (array-like, optional): Sample weights for each data point. Defaults to None.
        test_size (float, optional): Fraction of the dataset to be used as test data. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducible results. Defaults to 42.
        num_epochs (int, optional): Maximum number of training epochs. Defaults to 50.
        batch_size (int, optional): Batch size for training. Defaults to 256.
        optimizer_fn_name (str, optional): Name of the optimizer to use. Defaults to "Adam".
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.
        use_scheduler (bool, optional): Whether to use a learning rate scheduler. Defaults to False.
        patience (int, optional): Number of epochs with no improvement before stopping training. Defaults to 10.
        dtype (torch.dtype, optional): The dtype to use for input tensors. Defaults to torch.float32.
    """
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
            
            # if self.use_scheduler and i >= 2:
            #     scheduler.step()
            # if self.use_scheduler:
            #     scheduler.step()

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