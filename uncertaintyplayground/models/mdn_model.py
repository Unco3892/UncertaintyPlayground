import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable

class MDN(nn.Module):
    """
    Mixed Density Network (MDN) model.

    This model represents a mixture density network for modeling and predicting multi-modal distributions.

    Args:
        input_dim (int): Number of predictors for the first layer of the nueral network.
        n_gaussians (int): Number of Gaussian components in the mixture.
        dense1_units (int): Number of neurons in the first dense layer. Default is 10.
        prediction_method (str): Method for predicting the output distribution. Options are:
                                 - 'max_weight_mean': Choose the component with the highest weight and return the mean.
                                 - 'max_weight_sample': Choose a component from the mixture and sample from it.
                                 - 'average_sample': Draw multiple samples and take the average.

    Attributes:
        z_h (nn.Sequential): Hidden layer of the neural network.
        z_pi (nn.Linear): Linear layer for predicting mixture weights.
        z_mu (nn.Linear): Linear layer for predicting Gaussian means.
        z_sigma (nn.Linear): Linear layer for predicting Gaussian standard deviations.
        prediction_method (str): Method for predicting the output distribution.
    """

    def __init__(self, input_dim, n_gaussians, dense1_units = 10, prediction_method='max_weight_sample'):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(input_dim,dense1_units),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(dense1_units, n_gaussians)
        self.z_mu = nn.Linear(dense1_units, n_gaussians)
        self.z_sigma = nn.Linear(dense1_units, n_gaussians)
        self.prediction_method = prediction_method

    def forward(self, x):
        """
        Forward pass of the MDN model.

        Computes the parameters (pi, mu, sigma) of the output distribution given the input.

        Args:
            x (tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            tuple: A tuple containing the predicted mixture weights, means, and standard deviations.
        """
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h), -1)
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))
        return pi, mu, sigma

    def sample(self, x, num_samples=100):
        """
        Generate samples from the output distribution given the input.

        Args:
            x (tensor): Input tensor of shape (batch_size, num_features).
            num_samples (int): Number of samples to generate. Default is 100.

        Returns:
            tensor: A tensor of shape (batch_size,) containing the generated samples.
        """
        pi, mu, sigma = self.forward(x)

        if self.prediction_method == 'max_weight_mean':
            # Choose component with the highest weight
            pis = torch.argmax(pi, dim=1)
            # Return the mean of the chosen component
            sample = mu[torch.arange(mu.size(0)), pis]

        elif self.prediction_method == 'max_weight_sample':
            # Choose component from the mixture
            categorical = torch.distributions.Categorical(pi)
            pis = list(categorical.sample().data)
            # Sample from the chosen component
            sample = Variable(sigma.data.new(sigma.size(0)).normal_())
            for i in range(sigma.size(0)):
                sample[i] = sample[i] * sigma[i, pis[i]] + mu[i, pis[i]]

        elif self.prediction_method == 'average_sample':
            # Draw multiple samples and take the average
            samples = []
            for _ in range(num_samples):
                # Choose component from the mixture
                categorical = torch.distributions.Categorical(pi)
                pis = list(categorical.sample().data)
                # Sample from the chosen component
                sample = Variable(sigma.data.new(sigma.size(0)).normal_())
                for i in range(sigma.size(0)):
                    sample[i] = sample[i] * sigma[i, pis[i]] + mu[i, pis[i]]
                samples.append(sample)
            sample = torch.mean(torch.stack(samples), dim=0)

        else:
            raise ValueError(f"Invalid prediction method: {self.prediction_method}")

        return sample


# Define the loss function to be used by the MDN model (in the next iterations, this can be moved to the model class)
def mdn_loss(y, mu, sigma, pi):
    """
    Compute the MDN loss.

    Calculates the negative log-likelihood of the target variable given the predicted parameters of the mixture.

    Args:
        y (tensor): Target tensor of shape (batch_size,).
        mu (tensor): Predicted means tensor of shape (batch_size, n_gaussians).
        sigma (tensor): Predicted standard deviations tensor of shape (batch_size, n_gaussians).
        pi (tensor): Predicted mixture weights tensor of shape (batch_size, n_gaussians).

    Returns:
        tensor: The computed loss.
    """
    m = Normal(loc=mu, scale=sigma)
    log_prob = m.log_prob(y.unsqueeze(1))
    log_mix = torch.log(pi) + log_prob
    return -torch.logsumexp(log_mix, dim=1).mean()
    