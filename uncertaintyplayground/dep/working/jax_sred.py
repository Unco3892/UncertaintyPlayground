import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
from jax import jit
from jax.config import config
from jaxutils import Dataset
import jaxkern as jk

import tensorflow_probability.substrates.jax as tfp

tfb = tfp.bijectors

import distrax as dx
import gpjax as gpx
from gpjax.config import get_global_config, reset_global_config
import numpy as np

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
key = jr.PRNGKey(123)

#------------------------------------
x = jnp.array(r.x_train)
y = jnp.array(r.y_train).reshape(-1, 1)
xtest = jnp.array(r.x_test)

n, input_dim = x.shape
num_inducing_points = 100

# Reshape y to be 2-dimensional
# y = y[:, jnp.newaxis]

D = Dataset(X=x, y=y)

#------------------------------------
# Define inducing points
z = jnp.linspace(-15.0, 15.0, num_inducing_points).reshape(-1, 1)
z = jnp.tile(z, (1, input_dim))

# fig, ax = plt.subplots(figsize=(12, 5))
# ax.plot(x[:, 0], y, "o", alpha=0.3)
# ax.plot(xtest[:, 0], f(xtest))
# [ax.axvline(x=z_i, color="black", alpha=0.3, linewidth=1) for z_i in z[:, 0]]
# plt.show()

#------------------------------------
# define the model parameters
likelihood = gpx.Gaussian(num_datapoints=n)
prior = gpx.Prior(kernel=jk.RBF())
p = prior * likelihood
q = gpx.VariationalGaussian(prior=prior, inducing_inputs=z)

#------------------------------------
# run the model
svgp = gpx.StochasticVI(posterior=p, variational_family=q)
negative_elbo = jit(svgp.elbo(D, negative=True))

#------------------------------------
# fit the model
reset_global_config()
parameter_state = gpx.initialise(svgp, key)
optimiser = ox.rmsprop(learning_rate=0.9)

inference_state = gpx.fit_batches(
    objective=negative_elbo,
    parameter_state=parameter_state,
    train_data=D,
    optax_optim=optimiser,
    num_iters=5000,
    key=jr.PRNGKey(42),
    batch_size=128,
)

learned_params, training_history = inference_state.unpack()


#----------------------
# Get the predictions in batches
batch_size = 500  # Adjust the batch size based on available memory
num_batches = int(jnp.ceil(xtest.shape[0] / batch_size))
y_pred = []

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    xtest_batch = xtest[start_idx:end_idx, :]
    
    latent_dist_batch = q(learned_params)(xtest_batch)
    predictive_dist_batch = likelihood(learned_params, latent_dist_batch)
    y_pred_batch = predictive_dist_batch.mean()
    y_pred.append(y_pred_batch)

# y_pred = jnp.vstack(y_pred)
preds = jnp.concatenate([pred.reshape(-1, 1) for pred in y_pred], axis=0)
preds = np.array(preds)

# latent_dist = q(learned_params)(xtest)
# predictive_dist = likelihood(learned_params, latent_dist)
# y_pred = predictive_dist.mean()

# Assuming you have ground-truth values in your R script as `y_test`
y_test = jnp.array(r.y_test).reshape(-1, 1)
# r.y_comp[r.a_n:len(r.y_comp)]


# y_comp[(a_n-1):len(y_comp)]


# # Calculate performance metrics
# r2 = r2_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)

# print(f"R^2 score: {r2:.4f}")
# print(f"Mean Squared Error: {mse:.4f}")
# print(f"Mean Absolute Error: {mae:.4f}")
