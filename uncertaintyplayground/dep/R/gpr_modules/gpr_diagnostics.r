library(tidyselect)
library(keras)
library(tfprobability)

#' Predict distribution from a model
#'
#' This function predicts the distribution for a given test data instance, using a trained model.
#'
#' @param model Trained model for prediction
#' @param test_data Test data for prediction
#' @param instance_index Index of test data instance to predict
#' @param n_samples Number of samples to draw from the predicted distribution (set as 500 due to challenges with exact replication)
#'
#' @return A tidy data frame with `value` column containing predicted distribution samples
#'
#' @importFrom tensorflow tf convert_to_tensor set_random_seed
#' @importFrom tibble as_tibble
#' @importFrom stats as.numeric
#' @importFrom ggplot2 geom_density labs theme_minimal
#' @importFrom dplyr as_array
#'
#' @examples
#' predict_distribution(my_model, my_test_data, 1)
#'
#' @export
predict_distribution <- function(model, test_data, instance_index, n_samples = 500) {
  # # Prepare input for the model
  n_reps <- 2
  # if (instance_index == nrow(test_data)) {
  #   instance_index <- instance_index - 1
  # }
  # select two instances because there was tensorflow error when only one instance was selected and could not resolve it
  # input_data <- as.matrix(test_data[instance_index:(instance_index + 1), 1:(ncol(test_data) - 1)])

  # alternatively, we can do the following:alpha
  input_data <- as.matrix(test_data[instance_index, 1:(ncol(test_data) - 1)])
  input_data <- matrix(rep(t(input_data),n_reps), ncol=ncol(input_data), byrow=TRUE)
  # this has an implication that the number of test inferences, influences your prediction which is very bad

  # Make predictions for the specified instances
  distributions <- model(tf$convert_to_tensor(input_data))

  # for more info on the functions, see
  # https://github.com/rstudio/tfprobability/blob/main/R/distribution-methods.R
  # Draw samples from the distribution
  tensorflow::set_random_seed(42)

  # samples <- distributions$sample(n_samples) %>%
  samples <- distributions %>% 
    tfd_sample(n_samples) %>%
    tf$squeeze() %>% 
    tf$transpose()
    # as.array()

  # Convert samples to a tidy format & select the first entry of the second dimension i.e. `instance_index`
  # samples_tidy <- tibble(value = as.numeric(samples[, 1, ]))
  samples_tidy <- tibble(value = as.numeric(tf$reshape(samples, shape = as.integer(c(n_reps * n_samples)))))

  # return the values
  return(samples_tidy)
}

#' Plot predicted distribution
#'
#' This function plots the predicted distribution returned by `predict_distribution`.
#'
#' @param df Tidy data frame containing predicted distribution samples
#'
#' @importFrom ggplot2 geom_density labs theme_minimal
#'
#' @examples
#' plot_predict_distribution(my_distribution)
#'
#' @export
plot_predict_distribution <- function(df) {
  ggplot(df, aes(x = value)) +
    geom_density(fill = "blue", alpha = 0.5) +
    labs(x = "Predicted Value", y = "Density", title = paste0("Distribution of Prediction for Instance ", 1)) +
    theme_minimal()
}

#' Calculate Gaussian Process Regression Confidence Intervals
#'
#' This function takes in true values and a distribution object and calculates the Gaussian Process Regression (GPR) confidence intervals for each test data point.
#'
#' @param y_true A numeric vector of true values for the test set
#' @param distributions A distribution object returned by a trained TensorFlow model
#' @param n_samples An integer indicating the number of samples to draw from the distribution
#'
#' @return A data frame with columns `observation`, `y_test`, `y_mean`, `y_lower`, and `y_upper`
#'
#' @importFrom tensorflow set_random_seed as matrix tfd_sample tf$transpose as apply quantile
#' @export
gpr_confidence_intervals <- function(y_true, distributions, n_samples = 1000) {
  # Set the random seed for reproducibility
  tensorflow::set_random_seed(42)
  # Draw samples from the VGP layer distributions
  y_samples <- distributions %>%
    tfd_sample(n_samples) %>%
    tf$squeeze() %>%
    tf$transpose() %>%
    as.matrix()

  # Calculate the mean and quantiles for each test data point
  y_mean <- apply(y_samples, 1, mean)
  y_lower <- apply(y_samples, 1, function(x) quantile(x, 0.025))
  y_upper <- apply(y_samples, 1, function(x) quantile(x, 0.975))

  # Create a dataframe for plotting
  plot_data <- data.frame(
    observation = 1:length(y_true),
    y_test = y_true,
    y_mean = y_mean,
    y_lower = y_lower,
    y_upper = y_upper
  )
  return(plot_data)
}

#' Plot Gaussian Process Regression Confidence Intervals
#'
#' This function plots the true values and the GPR mean and confidence intervals for a specified number of test data points.
#'
#' @param df A data frame object returned by `gpr_confidence_intervals`
#' @param n_instances An integer indicating the number of test data points to plot
#'
#' @return A ggplot object
#'
#' @importFrom ggplot2 ggplot geom_point geom_ribbon scale_color_manual scale_shape_manual scale_fill_manual labs theme_gray element_blank
#' @export
plot_gpr_confidence_intervals <- function(df, n_instances = 100) {
  # Create a dataframe with the required columns
  data <- df
  data$outside_ci <- (data$y_test < data$y_lower) | (data$y_test > data$y_upper)
  data <- data[1:n_instances, ]

  # Plot
  ggplot(data = data, aes(x = observation)) +
    geom_point(aes(y = y_test, color = "True Value", shape = "True Value")) +
    geom_point(aes(y = y_mean, color = "Prediction", shape = "Prediction")) +
    geom_ribbon(aes(ymin = y_lower, ymax = y_upper, fill = "Confidence Interval"), alpha = 0.2) +
    scale_color_manual(name = "Legend", values = c("True Value" = "blue", "Prediction" = "green")) +
    scale_shape_manual(name = "Legend", values = c("True Value" = 16, "Prediction" = 17)) +
    scale_fill_manual(name = "Legend", values = c("Confidence Interval" = "red")) +
    labs(
      title = "Predictions with Confidence Intervals",
      x = "Observation",
      y = "Value"
    ) +
    theme_gray() +
    theme(legend.position = "top", legend.title = element_blank())
}

#' Plots the mean prediction and the 95% confidence interval for each test instance.
#'
#' @param df A data frame with the following columns:
#'   \item{y_true}{A numeric vector of the true target values.}
#'   \item{distributions}{A list of the output distributions of the GPR model.}
#'   \item{n_samples}{An integer specifying the number of samples to draw from the output distributions.}
#'
#' @return A ggplot object.
#'
#' @import ggplot2
#' @importFrom dplyr as_data_frame
#' @importFrom tidyr gather
#' @importFrom stats cor
#' @import tensorflow
#' @import tfprobability
#' @export
plot_gpr_mean_errorbars <- function(df) {
  # Calculate the correlation coefficient between the true and predicted values
  corr <- cor(df$y_test, df$y_mean)

  # Plot the data with error bars
  p <- ggplot(df, aes(x = y_test)) +
    geom_point(aes(y = y_mean), color = "blue", alpha = 0.6) +
    geom_errorbar(aes(ymin = y_lower, ymax = y_upper), width = 0.1, color = "red", alpha = 0.6) +
    labs(
      title = "Gaussian Process Regression Confidence Intervals",
      x = "True Target Values",
      y = "Predicted Mean and Confidence Intervals"
    ) +
    theme_minimal() +
    geom_abline(intercept = 0, slope = 1, color = "#168664") +
    annotate("text", x = max(df$y_test), y = max(df$y_mean), 
             label = paste("cor =", round(corr, 2)), hjust = 1, vjust = 1)

  # Return the plot
  return(p)
}

#' Plot Gaussian Process Regression densities for selected predictions.
#'
#' This function generates density plots for the predicted values of a specified
#' subset of instances, given a set of Gaussian Process Regression (GPR) output
#' distributions. The densities are computed from samples generated from the
#' GPR output distributions using TensorFlow Probability functions.
#'
#' @param y_distributions Output distributions from the GPR model.
#' @param indices Vector of indices of instances to plot densities for.
#' @param n_samples Number of samples to draw from the output distributions.
#' @return A ggplot2 density plot showing the densities of predicted values for
#' the selected instances.
#' @importFrom ggplot2 aes geom_density labs theme_minimal scale_fill_discrete
#' @importFrom tibble data_frame
#' @importFrom tensorflow tfd_sample tf$squeeze() tf$transpose()
#' @examples
#' \dontrun{
#' # Create example input data
#' input_data <- data.frame(x = seq(0, 10, length.out = 100))
#' output_data <- sin(input_data$x) + rnorm(n = 100, mean = 0, sd = 0.1)
#'
#' # Train a Gaussian Process Regression model
#' gpr_model <- gpr(input_data, output_data, kernel = "rbf", epsilon = 0.1)
#'
#' # Make predictions on test data and extract output distributions
#' test_data <- data.frame(x = seq(0, 10, length.out = 50))
#' gpr_preds <- predict(gpr_model, test_data, output_distribution = TRUE)
#' y_distributions <- gpr_preds$output_distributions
#'
#' # Plot densities for the first five instances
#' plot_gpr_densities(y_distributions, 1:5)
#' }
plot_gpr_densities <- function(y_distributions, indices, n_samples = 1000) {
  # Generate samples
  y_samples <- y_distributions %>%
    tfd_sample(n_samples) %>%
    tf$squeeze() %>%
    tf$transpose() %>%
    as.matrix()

  # Extract samples for the specified indices
  selected_samples <- y_samples[indices, , drop = FALSE]

  # Create a long format dataframe for ggplot
  plot_data <- data.frame(
    index = factor(rep(indices, each = n_samples)),
    value = as.vector(t(selected_samples))
  )
  # Plot density
  ggplot(plot_data, aes(x = value, fill = index)) +
    geom_density(alpha = 0.6) +
    labs(title = "Density Plot for Selected Predictions", x = "Value", y = "Density") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5), legend.position = "top") +
    scale_fill_discrete(name = "Index")
}
