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
from torch.autograd import Variable

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

        # Choose component from the mixture
        categorical = torch.distributions.Categorical(pi)
        pis = list(categorical.sample().data)

        sample = Variable(sigma.data.new(sigma.size(0)).normal_())
        for i in range(sigma.size(0)):
            sample[i] = sample[i]*sigma[i,pis[i]] + mu[i,pis[i]]
        return sample


# def mdn_loss(y, mu, sigma, pi):
#     m = Normal(loc=mu, scale=sigma)
#     loss = -torch.sum(pi * m.log_prob(y.unsqueeze(1)))
#     return loss / y.size(0)

def mdn_loss(y, mu, sigma, pi):
    m = Normal(loc=mu, scale=sigma)
    log_prob = m.log_prob(y.unsqueeze(1))
    log_mix = torch.log(pi) + log_prob
    return -torch.logsumexp(log_mix, dim=1).mean()

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
        # self.model = self.model.double()  # convert model parameters to float64
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


#----------------- Genrating some data to test the method -----------------#
# # Example usage of the MDNTrainer class
# def generate_data(num_samples, mixture_weights, means, std_devs):
#     # Decide the number of samples from each component
#     num_samples_components = np.random.multinomial(num_samples, mixture_weights)
    
#     # Generate samples from each component
#     samples = []
#     for i, (num, mean, std_dev) in enumerate(zip(num_samples_components, means, std_devs)):
#         samples.extend(np.random.normal(mean, std_dev, num))
    
#     return np.array(samples)

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# # Assuming that your data is numpy arrays
# # Generate bimodal data
# num_samples = 10000
# mixture_weights = [0.9, 0.1]  # Adjust the weights to get bimodal distribution
# means = [-3.0, 3.0]  # Means for the two modes
# std_devs = [0.5, 0.5]  # Standard deviations for the two modes

# X = np.random.rand(num_samples, 20)  # 20 random input features
# y = generate_data(num_samples, mixture_weights, means, std_devs)

#----------------- Generate mixture data -----------------#
import math

def generate_mixed_data(num_samples, mixture_weights, means, std_devs, normal_mean, normal_std_dev):
    # Decide the number of samples from each component
    num_samples_components = np.random.multinomial(math.floor(num_samples / 2), mixture_weights)
    
    # Generate samples from each component
    samples = []
    for i, (num, mean, std_dev) in enumerate(zip(num_samples_components, means, std_devs)):
        samples.extend(np.random.normal(mean, std_dev, num))

    # Generate samples from normal distribution
    normal_samples = np.random.normal(normal_mean, normal_std_dev, math.floor(num_samples / 2))
    
    # Concatenate the samples
    all_samples = np.concatenate((samples, normal_samples))

    return all_samples

# Generate bimodal data
num_samples = 10000
mixture_weights = [0.9, 0.1]  # Adjust the weights to get bimodal distribution
means = [-3.0, 3.0]  # Means for the two modes
std_devs = [0.5, 0.5]  # Standard deviations for the two modes
normal_mean = 0.0  # Mean for the normal distribution
normal_std_dev = 1.0  # Standard deviation for the normal distribution

X = np.random.rand(num_samples, 20)  # 20 random input features
y = generate_mixed_data(num_samples, mixture_weights, means, std_devs, normal_mean, normal_std_dev)

#----------------- Training the MDN -----------------#
# # Initialize and train the MDN trainer
# trainer = MDNTrainer(X, y, num_epochs=100, lr=0.01, n_hidden=20, n_gaussians=4)
# def __init__(self, X, y, sample_weights=None, test_size=0.2, random_state=42, num_epochs=50, batch_size=256, optimizer_fn_name="Adam", lr=0.01, use_scheduler=False, patience=10, dtype=torch.float32):
trainer = MDNTrainer(X, y, num_epochs=100, lr=0.001, patience=20, n_hidden=20, n_gaussians=4,dtype=torch.float32)
trainer.train()

#----------------- Plotting the results -----------------#

import seaborn as sns
import matplotlib.pyplot as plt

def compare_distributions(trainer, x_instance, y_actual=None, num_samples=10000):
    """
    Compare the actual and predicted outcome distributions.

    Args:
        trainer (MDNTrainer): The trained MDNTrainer instance.
        x_instance (np.ndarray): The instance for which to predict the outcome distribution.
        y_actual (float or np.ndarray, optional): The actual outcome(s). If a single value, plot as a vertical line.
                                                  If an array or list, plot as a KDE. If None, don't plot actual outcome.
        num_samples (int, optional): The number of samples to generate from the predicted and actual distributions.
                                     Default is 10000.
    """
    # Ensure x_instance is a 2D array
    if x_instance.ndim == 1:
        x_instance = np.expand_dims(x_instance, axis=0)
    
    # Get the predicted parameters of the mixture distribution
    pi, mu, sigma, pred = trainer.predict_with_uncertainty(x_instance)
    
    # Generate samples from the predicted distribution
    predicted_samples = []
    
    print ("The weights are: ",pi, "\n", 
           "The means are: ",mu, "\n",
           "The sigmas are: ",sigma, "\n",
           "The final prediction is: ",pred, "\n")

    for _ in range(num_samples):
        # Choose Gaussian
        idx = np.random.choice(np.arange(len(pi[0])), p=pi[0])
        # Sample from Gaussian
        sample = np.random.normal(mu[0, idx], sigma[0, idx])
        predicted_samples.append(sample)
    
    # Plot KDE of predicted samples
    sns.kdeplot(predicted_samples, fill=True, color="r", label="Predicted distribution")

    # If y_actual is provided
    if y_actual is not None:
        # If y_actual is a single value, plot it as a vertical line
        if np.isscalar(y_actual):
            plt.axvline(y_actual, color='b', linestyle='--', label='Ground truth')
        else:
            # If y_actual is a list or array, plot it as a KDE
            sns.kdeplot(y_actual, fill=True, color="b", label="Actual")
        plt.title('Outcome Distributions: Actual vs Predicted')
    else:
        plt.title('Outcome Distributions: Predicted')
        
    # plot the predicted value
    plt.axvline(pred.numpy(), color='r', linestyle='--', label='Predicted mean')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.show()

# Generate some testing data
X_test = np.random.rand(1000, 20)
# Call the function with the actual distribution
#Y_test = generate_mixed_data(num_samples, mixture_weights, means, std_devs)
Y_test = generate_mixed_data(num_samples, mixture_weights, means, std_devs, normal_mean, normal_std_dev)

# Choose a test instance
index_instance = 900
test_instance = X_test[index_instance, :]
test_instance = test_instance.astype(np.float64)

print(test_instance)

# Get the predicted parameters of the mixture distribution
pi, mu, sigma, _ = trainer.predict_with_uncertainty(test_instance)

# print(test_instance.shape)
# # Call the function with the actual value
# compare_distributions(trainer, test_instance)

compare_distributions(trainer, test_instance, y_actual=Y_test[index_instance])
