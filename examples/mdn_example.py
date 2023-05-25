import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import r2_score

class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(20, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h), -1)
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))
        return pi, mu, sigma

    def sample(self, x):
        pi, mu, sigma = self.forward(x)
        pi = pi.detach().cpu().numpy()  # Shape: (n, K)
        mu = mu.detach().cpu().numpy()  # Shape: (n, K)
        sigma = sigma.detach().cpu().numpy()  # Shape: (n, K)

        # Pick component from the mixture
        k = np.array([np.random.choice(len(p), p=p) for p in pi])
        return np.array([np.random.normal(mu_i[k_i], sigma_i[k_i]) for mu_i, sigma_i, k_i in zip(mu, sigma, k)])


def mdn_loss(y, mu, sigma, pi):
    m = Normal(loc=mu, scale=sigma)
    loss = -torch.sum(pi * m.log_prob(y.unsqueeze(1)))
    return loss / y.size(0)

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

class MDNTrainer(BaseTrainer):
    def __init__(self, *args, n_hidden=20, n_gaussians=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gaussians = n_gaussians

        self.model = MDN(n_hidden=n_hidden, n_gaussians=self.n_gaussians).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def train(self):
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

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {loss.item():.3f}, Validation Loss: {val_loss.item():.3f}")

            should_stop = early_stopping(val_loss.item(),            self.model)

            if should_stop:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        if early_stopping.best_model_state is not None:
            self.model.load_state_dict(early_stopping.best_model_state)
            self.model.eval()

    def predict_with_uncertainty(self, X):
        """
        Predicts the output distribution given input tensor X using the trained MDN.

        Args:
            X (tensor): Input tensor of shape (num_samples, num_features).

        Returns:
            tuple: A tuple containing the output distribution's parameters (pi, mu, sigma) and a sample from the distribution.
        """
        self.model.eval()

        # Convert numpy array to PyTorch tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)

        # Check if X is a single instance and add an extra dimension if necessary
        if X.ndim == 1:
            X = torch.unsqueeze(X, 0)

        with torch.no_grad():
            pi, mu, sigma = self.model(X)
            sample = self.model.sample(X)

        return pi.cpu().numpy(), mu.cpu().numpy(), sigma.cpu().numpy(), sample


# Example usage of the MDNTrainer class

# Assuming that your data is numpy arrays
X = np.random.rand(1000, 20)
y = np.random.rand(1000)

# we also set the seeds for the models
torch.manual_seed(42)
np.random.seed(42)

# Create an instance of the MDNTrainer class
trainer = MDNTrainer(X=X, y=y, n_hidden=20, n_gaussians=5, num_epochs=5000, batch_size=1000, lr=0.2, patience=3)

# Train the model
trainer.train()

# Make predictions with uncertainty
pi, mu, sigma, y_pred = trainer.predict_with_uncertainty(trainer.X_val.numpy())

#print(pi,mu,sigma)

# Compute R2 score for validation set
print('R2 Score:', r2_score(trainer.y_val, y_pred))

# ------------------------------
# plot the prediction for one instance
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# # Assume you want to see the distribution for the first sample in your validation set
# sample_idx = 0

# # Make prediction for a single instance
# instance = trainer.X_val[sample_idx].numpy()
# sample_pi, sample_mu, sample_sigma, y_pred = trainer.predict_with_uncertainty(instance)

# # # Get the parameters for the mixture model for this sample
# # sample_pi = pi[sample_idx]
# # sample_mu = mu[sample_idx]
# # sample_sigma = sigma[sample_idx]

# # Generate a range of x values
# x_values = np.linspace(-10, 10, 400)

# # Compute the total density function
# total_density = np.zeros_like(x_values)
# for i in range(len(sample_pi)):
#     density = sample_pi[i] * norm.pdf(x_values, sample_mu[i], sample_sigma[i])
#     total_density += density
#     plt.plot(x_values, density, label=f'Component {i+1}')

# # Plot the total density function
# plt.plot(x_values, total_density, label='Total', linestyle='--')
# plt.legend()
# plt.show()

import matplotlib.pyplot as plt
from scipy.stats import norm

# Let's assume you want to see the distribution for the first sample in your validation set
def plot_mixture_model(instance_num, pi, mu, sigma):
    # Get the parameters for the mixture model for this instance
    sample_pi = pi[instance_num]
    sample_mu = mu[instance_num]
    sample_sigma = sigma[instance_num]

    # Generate a range of x values
    x_values = np.linspace(min(sample_mu - 3*sample_sigma), max(sample_mu + 3*sample_sigma), 400)

    # Compute the total density function
    total_density = np.zeros_like(x_values)
    for i in range(len(sample_pi)):
        density = sample_pi[i] * norm.pdf(x_values, sample_mu[i], sample_sigma[i])
        total_density += density
        plt.plot(x_values, density, label=f'Component {i+1}')

    # Plot the total density function
    plt.plot(x_values, total_density, label='Total', linestyle='--')
    plt.legend()
    plt.title(f'Instance {instance_num} Mixture Model')
    plt.show()


## Plot the mixture model for instance number 0
# plot_mixture_model(0, pi, mu, sigma)

#------------------------
# trying my model with non-normally distributed data
def generate_data(num_samples, mixture_weights, means, std_devs):
    # Decide the number of samples from each component
    num_samples_components = np.random.multinomial(num_samples, mixture_weights)
    
    # Generate samples from each component
    samples = []
    for i, (num, mean, std_dev) in enumerate(zip(num_samples_components, means, std_devs)):
        samples.extend(np.random.normal(mean, std_dev, num))
    
    return np.array(samples)

# Generate the data
num_samples = 10000
data = generate_data(num_samples, mixture_weights, means, std_devs)

# Parameters of the Gaussian components
mixture_weights = [0.3, 0.7]
means = [0.0, 5.0]
std_devs = [1.0, 0.5]

# Generate the data
num_samples = 10000
X = np.random.rand(num_samples, 20)
y = generate_data(num_samples, mixture_weights, means, std_devs)

# we also set the seeds for the models
torch.manual_seed(42)
np.random.seed(42)

# Create an instance of the MDNTrainer class
trainer = MDNTrainer(X=X, y=y, n_hidden=20, n_gaussians=5, num_epochs=5000, batch_size=1000, lr=0.01, patience=3)

# Train the model
trainer.train()

# Make predictions with uncertainty
pi, mu, sigma, y_pred = trainer.predict_with_uncertainty(trainer.X_val.numpy())

# Compute R2 score for validation set
print('R2 Score:', r2_score(trainer.y_val, y_pred))

# Plot the mixture model for instance number 0
plot_mixture_model(0, pi, mu, sigma)