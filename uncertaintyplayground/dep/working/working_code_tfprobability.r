library(keras)
library(tensorflow)
library(tfprobability)

# Creating the data
a_n <- 1500
x_train <- hidden_node[1:a_n,]
y_train <- y_comp[1:a_n]
x_test <- hidden_node[a_n:nrow(y_comp),]

# Define the kernel
amplitude <- tf$Variable(initial_value = 1.0, dtype = tf$float64, name = "amplitude")
length_scale <- tf$Variable(initial_value = 1.0, dtype = tf$float64, name = "length_scale")
kernel <- tfp$math$psd_kernels$ExponentiatedQuadratic(amplitude, length_scale)

# Create the Gaussian Process
observation_noise_variance <- tf$Variable(initial_value = 1e-6, dtype = tf$float64, name = "observation_noise_variance")
gp = tfp$distributions$GaussianProcess(kernel, index_points = x_train, observation_noise_variance = observation_noise_variance)

# Create the Gaussian Process Regression model
gprm = tfp$distributions$GaussianProcessRegressionModel(kernel, x_test, x_train, y_train, observation_noise_variance)

# Fit the model
negative_log_likelihood <- function() -gp$log_prob(y_train)
optimizer <- tf$optimizers$Adam()
for (i in 1:100) {
  optimizer$minimize(negative_log_likelihood, var_list = list(amplitude, length_scale, observation_noise_variance))
}

# Make predictions
mean_predictions <- gprm$mean()
stddev_predictions <- gprm$stddev()

# Convert to R arrays
mean_predictions <- as.array(mean_predictions)
stddev_predictions <- as.array(stddev_predictions)

x_train %>% str()
y_train %>% str()
x_test %>% str()
y_test[,1]  %>% str()


to_pred <- a_n:nrow(y_comp)
preds <- gp$predict(hidden_node[to_pred,], return_std=T) %>% py_to_r()
y_preds <- preds[[1]]
y_std <- preds[[2]]
# Evaluate the model
postResample(pred = y_preds, obs = y_comp[to_pred])

#--------------------------------
# Define the kernel
amplitude <- tf$Variable(initial_value = 1.0, dtype = tf$float64, name = "amplitude")
length_scale <- tf$Variable(initial_value = 1.0, dtype = tf$float64, name = "length_scale")
kernel <- tfp$math$psd_kernels$ExponentiatedQuadratic(amplitude, length_scale)

# Create the Gaussian Process
observation_noise_variance <- tf$Variable(initial_value = 1e-6, dtype = tf$float64, name = "observation_noise_variance")
gp = tfp$distributions$GaussianProcess(kernel, index_points = x_train, observation_noise_variance = observation_noise_variance)

# Create the Gaussian Process Regression model
gprm = tfp$distributions$GaussianProcessRegressionModel(kernel, x_test, x_train, y_train, observation_noise_variance)

# Fit the model
negative_log_likelihood <- function() -gp$log_prob(y_train)
optimizer <- tf$optimizers$Adam(learning_rate = 0.5)
iterations <- 138

for (i in 1:iterations) {
  optimizer$minimize(negative_log_likelihood, var_list = list(amplitude, length_scale, observation_noise_variance))
  message(paste0("It", i, ": ", negative_log_likelihood()))
}

# Make predictions
mean_predictions <- gprm$mean()
stddev_predictions <- gprm$stddev()

# Convert to R arrays
mean_predictions <- as.array(mean_predictions)
stddev_predictions <- as.array(stddev_predictions)

postResample(pred = as.vector(mean_predictions), obs = y_comp[a_n:nrow(y_comp)])

#------------------------------------------------------------------------------