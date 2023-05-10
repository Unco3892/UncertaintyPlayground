import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from flax import linen as nn
from jax.scipy.stats import multivariate_normal

class SparseGPR(nn.Module):
    num_inducing_points: int

    def setup(self):
        self.inducing_points = self.param('inducing_points', (self.num_inducing_points, 50), nn.initializers.normal())

    def __call__(self, x, y, noise_variance):
        # Kernel function (RBF)
        def rbf_kernel(x, y, length_scale=1.0, amplitude=1.0):
            diff = x - y
            return amplitude * jnp.exp(-0.5 * jnp.sum(diff ** 2) / length_scale ** 2)

        kernel = vmap(vmap(rbf_kernel, (0, None), 0), (None, 0), 0)

        # Covariance matrices
        Kuu = kernel(self.inducing_points, self.inducing_points)
        Kuf = kernel(self.inducing_points, x)
        Kfu = jnp.transpose(Kuf)
        Kff = kernel(x, x)

        # Calculate posterior mean and covariance
        L = jnp.linalg.cholesky(Kuu + 1e-6 * jnp.eye(self.num_inducing_points))
        A = jnp.linalg.solve(L, Kuf)
        V = jnp.linalg.solve(L, A)

        mu = jnp.matmul(jnp.transpose(A), jnp.linalg.solve(L, Kfu @ y))
        cov = Kff - jnp.matmul(jnp.transpose(A), A) + noise_variance * jnp.eye(Kff.shape[0])

        return mu, cov

# Training and prediction
def train_and_predict(sparse_gpr, x_train, y_train, x_test, noise_variance):
    return sparse_gpr(x_test, y_train, noise_variance)

key = random.PRNGKey(0)
num_inducing_points = 500
num_observations = 20000
num_predictors = 50

# Generate synthetic data
x_train = random.normal(key, (num_observations, num_predictors))
y_train = random.normal(key, (num_observations, 1))
x_test = random.normal(key, (100, num_predictors))

# Initialize and train Sparse GPR
sparse_gpr = SparseGPR(num_inducing_points)
params = sparse_gpr.init(key, x_train, y_train, 1.0)
y_pred_mu, y_pred_cov = train_and_predict(sparse_gpr, x_train, y_train, x_test, 1.0)

print("Predicted mean:", y_pred_mu)
print("Predicted covariance:", y_pred_cov)
