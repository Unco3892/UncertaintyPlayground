library(keras)
library(tensorflow)
library(tfprobability)

# OUR TEST SET IS ACTUALLY A VALIDATION SET!

# generate random data
# set seed for reproducibility
# set.seed(123)
# # number of features for x
# n_features <- 50
# # total number of instances
# n_instances <- 25000
# # create simulated x_train data
# df <- matrix(rnorm(n_features * n_instances), nrow = n_instances)
# # create simulated y_train data
# y_comp <- matrix(rnorm(n_instances))

# split x_train data into training and test sets
a_n <- 1500
# x_train <- df[1:a_n,]
# y_train <- y_comp[1:a_n]
# x_test <- df[a_n:nrow(y_comp),]
x_train <- hidden_node[1:a_n,]
y_train <- y_comp[1:a_n]
x_test <- hidden_node[a_n:nrow(y_comp),]

# Convert y_train to a float64 tensor
y_train <- tf$cast(y_train, tf$float64)

# Define the kernel
amplitude <- tf$Variable(initial_value = 1.0, dtype = tf$float64, name = "amplitude")
length_scale <- tf$Variable(initial_value = 1.0, dtype = tf$float64, name = "length_scale")
kernel <- tfp$math$psd_kernels$ExponentiatedQuadratic(amplitude, length_scale)
observation_noise_variance <- tf$Variable(initial_value = 1e-4, dtype = tf$float64, name = "observation_noise_variance")


# Define inducing points
num_inducing_points <- 200L
inducing_index_points <- x_train[seq(1, nrow(x_train), length.out = num_inducing_points),]

# Define variational parameters
variational_inducing_observations_loc <- tf$Variable(initial_value = tf$zeros(shape = num_inducing_points, dtype = tf$float64), name = "variational_inducing_observations_loc")
variational_inducing_observations_scale <- tf$Variable(initial_value = tf$eye(num_inducing_points, dtype = tf$float64), name = "variational_inducing_observations_scale")

# Create the VariationalGaussianProcess model
vgp = tfp$distributions$VariationalGaussianProcess(
  kernel = kernel,
  index_points = x_train,
  inducing_index_points = inducing_index_points,
  observation_noise_variance = observation_noise_variance,
  variational_inducing_observations_loc = variational_inducing_observations_loc,
  variational_inducing_observations_scale = variational_inducing_observations_scale
)


# Fit the model
jitter <- 1e-3
negative_log_likelihood <- tf_function(function() -vgp$log_prob(y_train))
optimizer <- tf$optimizers$RMSprop(learning_rate = 0.99)
# optimizer <- tf$optimizers$Adam(learning_rate = 1.0)
# optimizer <- tf$optimizers$Adam(learning_rate = learning_rate_schedule)
iterations <- 500

# Set the number of iterations with no improvement before stopping
n_no_improvement <- 1

# Initialize the counter for no improvement
no_improvement_count <- 0

# Initialize the best MSE and the corresponding model parameters
best_mse = Inf
best_params = NULL

# we can also get the model parameters before the training
cat("Amplitude before training:", as.array(amplitude), "\n")
cat("Length scale before training:", as.array(length_scale), "\n")
cat("Observation noise variance before training:", as.array(observation_noise_variance), "\n")

# Set an initial MSE value
prev_mse <- Inf

start_time <- Sys.time()
for (i in 1:iterations) {
  optimizer$minimize(negative_log_likelihood, var_list = list(amplitude, length_scale, observation_noise_variance, variational_inducing_observations_loc, variational_inducing_observations_scale))
  
  # Add jitter to observation_noise_variance
  observation_noise_variance$assign(observation_noise_variance + jitter)
  
  # Create the Gaussian Process Regression model
  gprm = tfp$distributions$GaussianProcessRegressionModel(kernel, x_test, x_train, y_train, observation_noise_variance)
  
  # Make predictions
  mean_predictions <- gprm$mean()
  
  # Convert to R arrays
  mean_predictions <- as.array(mean_predictions)

  # Check for NaN values in negative log-likelihood
  i_log_likelihood <- negative_log_likelihood()$numpy()

  if (is.nan(i_log_likelihood)) {
    message("Stopping training because negative log-likelihood is NaN")
    message("Traning stopped at iteration ", i)
    break
  }

  # Calculate MSE and R2
  mse <- mean((y_comp[a_n:nrow(y_comp)] - mean_predictions)^2)
  r2 <- 1 - sum((y_comp[a_n:nrow(y_comp)] - mean_predictions)^2) / sum((y_comp[a_n:nrow(y_comp)] - mean(y_comp[a_n:nrow(y_comp)]))^2)

  # Check if MSE is not improving
  if (mse > prev_mse) {
    no_improvement_count = no_improvement_count + 1
  } else {
    no_improvement_count = 0
    prev_mse = mse
    # Check if this is the best MSE so far
    if (mse < best_mse) {
      best_mse <- mse
      # Save the model parameters at this iteration
      best_params = list(
        amplitude = as.array(amplitude),
        length_scale = as.array(length_scale),
        observation_noise_variance = as.array(observation_noise_variance),
        variational_inducing_observations_loc = as.array(variational_inducing_observations_loc),
        variational_inducing_observations_scale = as.array(variational_inducing_observations_scale)
      )
    }
  }

  # Check if we have reached the maximum number of iterations with no improvement
  if (no_improvement_count >= n_no_improvement) {
    message("Stopping training because MSE is not improving")
    message("Training stopped at iteration ", i)
    break
  }

  # Update prev_mse
  prev_mse <- mse

  message(paste0("Iteration ", i, ": Negative Log-Likelihood = ", round(i_log_likelihood,2), ", MSE = ", round(mse,4), ", R2 = ", round(r2,3)))
}
end_time <- Sys.time()
end_time - start_time

# Recover the model parameters from the best iteration
amplitude = tf$Variable(initial_value = best_params$amplitude, dtype = tf$float64, name = "amplitude")
length_scale = tf$Variable(initial_value = best_params$length_scale, dtype = tf$float64, name = "length_scale")
observation_noise_variance = tf$Variable(initial_value = best_params$observation_noise_variance, dtype = tf$float64, name = "observation_noise_variance")
variational_inducing_observations_loc = tf$Variable(initial_value = best_params$variational_inducing_observations_loc, dtype = tf$float64, name = "variational_inducing_observations_loc")
variational_inducing_observations_scale = tf$Variable(initial_value = best_params$variational_inducing_observations_scale, dtype = tf$float64, name = "variational_inducing_observations_scale")

# we can also get the model parameters
cat("Amplitude after training:", as.array(amplitude), "\n")
cat("Length scale after training:", as.array(length_scale), "\n")
cat("Observation noise variance after training:", as.array(observation_noise_variance), "\n")

# Add also the validation criteria
# Make a function to predict the values


# Make final predictions
mean_predictions <- gprm$mean()
# stddev_predictions <- gprm$stddev()

# Convert to R arrays
mean_predictions <- as.array(mean_predictions)
# stddev_predictions <- as.array(stddev_predictions)

# Use postResample
postResample(pred = as.vector(mean_predictions), obs = y_comp[a_n:nrow(y_comp)])


# predictios with the jax model
#pp <- py$preds %>% as.vector()
#postResample(pred = pp, obs = y_comp[a_n:nrow(y_comp)])

#-------------------#
# Added a new functio nfor doing inference
predict_gp <- function(new_instance) {
  # Ensure the new instance is a matrix with a single row
  new_instance <- matrix(new_instance, nrow = 1)

  # Create the Gaussian Process Regression model for the new instance
  gprm_new_instance <- tfp$distributions$GaussianProcessRegressionModel(
    kernel,
    new_instance,
    x_train,
    y_train,
    observation_noise_variance
  )

  # Make a prediction for mean
  mean_prediction <- gprm_new_instance$mean()
  # Make a prediction for std
  std_prediction <- gprm_new_instance$stddev()

  # Convert the prediction to an R array
  mean_prediction <- as.array(mean_prediction)

  # Convert the prediction to an R array
  std_prediction <- as.array(std_prediction)

  # return(list(mean_pred = mean_prediction))
  return(list(mean_pred = mean_prediction, std_pred = std_prediction))
}

# Create a new instance
# new_instance <- rnorm(50L)
new_instance <- x_test[10,]

# Make a prediction using the predict_gp function
prediction <- predict_gp(new_instance)

cat("Mean prediction for the new instance:", prediction[[1]], "\n")
cat("Std prediction for the new instance:", prediction[[2]], "\n")

# Then use map to make the prediction

# Plot the distribution of the predictions (compute the mean & std)

#-------------------#
# In the `tfp$distributions$GaussianProcessRegressionModel` function, we use both `x_test` and `x_train` because the function constructs a distribution over the function values at the test points `x_test` conditioned on the observed function values `y_train` at the training points `x_train`. In other words, the model leverages the information from the training data to make predictions at the test points.

# The Gaussian Process Regression Model function takes the following parameters:

# 1. `kernel`: The kernel function used in the Gaussian process. It specifies the covariance between any two points in the function space.
# 2. `index_points`: The input points (test points) at which we want to make predictions.
# 3. `observation_index_points`: The input points (training points) at which the function values have been observed.
# 4. `observations`: The observed function values (target values) at the training points.
# 5. `observation_noise_variance`: The noise variance added to the diagonal of the covariance matrix.
# In the code, `x_test` is passed as `index_points`, and `x_train` is passed as `observation_index_points`. The Gaussian Process Regression Model (GPRM) will use the kernel function along with the training points (`x_train`, `y_train`) and their associated observation noise variance to make predictions at the test points (`x_test`).

# > The inducing points are taken from the training set or the test set?
# Inducing points are selected from the training set. They are a subset of the training points used to approximate the full Gaussian process. The use of inducing points helps reduce the computational complexity of the Gaussian process model, making it more efficient for large datasets.
# In the provided code, inducing points are selected from the `x_train` set:
# ```
# # Define inducing points
# num_inducing_points <- 50L
# inducing_index_points <- x_train[seq(1, nrow(x_train), length.out = num_inducing_points),]
# ```
# Here, `num_inducing_points` is set to 50, and `inducing_index_points` are chosen by selecting points from `x_train` at equally spaced indices. This selection method is simple but might not always result in the best approximation. More advanced methods, such as K-means clustering or greedy selection, can be used to choose more representative inducing points.

#-------------------#
# Here's an explanation of the three parameters in the Gaussian Process Regression example:
# Amplitude and length scale of the kernel function:
# 1. The kernel function defines the covariance between any two points in the function space. In this example, the Exponentiated Quadratic kernel (also known as the Radial Basis Function or Squared Exponential kernel) is used. It has two parameters:
  ## Amplitude: This parameter controls the vertical scaling of the kernel function. A higher amplitude leads to greater variability in the functions represented by the Gaussian Process.
  ## Length scale: This parameter controls the horizontal scaling or the "smoothness" of the kernel function. A smaller length scale makes the functions represented by the Gaussian Process vary more rapidly, whereas a larger length scale results in smoother functions.
# 2. Observation noise variance: The observation noise variance is a parameter that accounts for the noise in the observed data (i.e., the target values). When using Gaussian Process Regression, we assume that the observed data is a noisy version of the true underlying function. The observation noise variance represents the magnitude of that noise. A higher value indicates a noisier observation, while a lower value indicates less noise in the data.
# 3. Variational inducing observations (mean and covariance): Inducing points are a subset of the training points used to approximate the full Gaussian process. They help reduce the computational complexity of the Gaussian process model, making it more efficient for large datasets. In the code, Variational Gaussian Process (VGP) is used, which is an approximation technique that leverages inducing points and optimizes the variational parameters associated with these points.
  ## Variational inducing observations location (mean): This represents the mean function values at the inducing points in the approximate Gaussian process. The optimizer tries to find the optimal mean function values that best approximate the full Gaussian process.
  ## Variational inducing observations scale (covariance): This represents the covariance matrix (or scale matrix) of the function values at the inducing points in the approximate Gaussian process. The optimizer tries to find the optimal covariance matrix that best captures the relationships between the inducing points and the full Gaussian process.

