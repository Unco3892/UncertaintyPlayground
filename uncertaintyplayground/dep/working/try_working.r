library(keras)
library(tensorflow)
library(tfprobability)

# generate random data
# set seed for reproducibility
set.seed(123)
# number of features for x
#n_features <- 50
# total number of instances
#n_instances <- 25000
# create simulated x_train data
#df <- matrix(rnorm(n_features * n_instances), nrow = n_instances)
# create simulated y_train data
#y_comp <- matrix(rnorm(n_instances))

# split x_train data into training and test sets
a_n <- 200
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
observation_noise_variance <- tf$Variable(initial_value = 1e-6, dtype = tf$float64, name = "observation_noise_variance")


# Define inducing points
num_inducing_points <- 50L
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

# define sample weights (optional)
# sample_weights <- NULL
# sample_weights <- NULL
sample_weights <- w_comp[1:a_n]
sample_weights <- tf$convert_to_tensor(sample_weights, dtype = tf$float64)
# sample_weights <- rep(1, a_n)
# sample_weights <- tf$convert_to_tensor(sample_weights, dtype = tf$float64)

# Fit the model
jitter <- 1e-6
negative_log_likelihood <- function() {
  # compute negative log-likelihood
  log_prob <- vgp$log_prob(y_train)
  # apply weights (if provided)
  if (!is.null(sample_weights)) {
    weighted_log_prob <- sample_weights * log_prob 
  } else {
    weighted_log_prob <- log_prob
  }
    # # print unweighted and weighted log-likelihood
    # message(paste0("Iteration ", i, ": Unweighted Log-Likelihood = ", -sum(log_prob), ", Weighted Log-Likelihood = ", -sum(weighted_log_prob)))

  # sum over instances
  sum(-weighted_log_prob)
}
optimizer <- tf$optimizers$Adam(learning_rate = 0.5)
iterations <- 200

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
  
  # Calculate MSE and R2
  mse <- mean((y_comp[a_n:nrow(y_comp)] - mean_predictions)^2)
  r2 <- 1 - sum((y_comp[a_n:nrow(y_comp)] - mean_predictions)^2) / sum((y_comp[a_n:nrow(y_comp)] - mean(y_comp[a_n:nrow(y_comp)]))^2)
  
  message(paste0("Iteration ", i, ": Negative Log-Likelihood = ", negative_log_likelihood(), ", MSE = ", mse, ", R2 = ", r2))
}

# Make predictions
mean_predictions <- gprm$mean()
sample_predictions <- gprm$sample(1)

# Convert to R arrays
mean_predictions <- as.array(mean_predictions)
sample_predictions <- as.array(sample_predictions)

# Plot predictive distribution
par(mfrow = c(1,2))
hist(mean_predictions, main = "Mean Predictions")
hist(sample_predictions, main = "Sampled Predictions")


# Generate predictions for selected instance
mean_pred <- gprm$mean()[instance_index,]
stddev_pred <- gprm$stddev()[instance_index,]


num_samples <- 10
sampled_predictions <- gprm$covariance(

# Plot histogram of sampled predictions
hist(sampled_predictions, main = "Predictive Distribution", xlab = "Prediction")

sample_predictions <- gprm$covariance()

sample_predictions %>% str()

# Make a single prediction for the first instance in the test set using the trained model
x_single <- x_test[1:2,]
tf$name_scope("prediction", {
  mean_prediction_single <- gprm$mean(x_single)
})

gpr$
