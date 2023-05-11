# interesting blog on variational convnets to get density estimates

library(tidyverse)
library(readxl)
library(rsample)
library(reticulate)
library(tfdatasets)
library(keras)
library(tfprobability)
library(purrr)
library(tensorflow)

# Suppress TensorFlow warnings
# tf$logging$set_verbosity(tf$logging$FATAL)
# tensorflow::tf$random$set_seed(42L)
tensorflow::set_random_seed(1)
tf$compat$v1$enable_eager_execution()

# Load the r2_score function from the gpr_utilities module
source(here::here("scripts/r/gpr_modules/gpr_utilities.r"))

# Load the gpr_kernels module
source(here::here("scripts/r/gpr_modules/gpr_kernels.r"))

#' Custom loss function for Variational Gaussian Process (VGP) regression
#'
#' This function defines a custom loss function for VGP regression using TensorFlow Probability.
#' It computes the variational loss of the target variable `y` and the random variable `rv_y`.
#'
#' @param y A tensor representing the true target values.
#' @param rv_y A random variable object from the VGP layer in the model.
#' @param kl_weight A scalar value determining the weight of the Kullback-Leibler (KL) divergence term in the loss function. Default is NULL.
#' @param sample_weights An optional tensor of the same shape as `y` representing the weights of each observation. Default is NULL.
#' Alternatively, you can also use the `sample_weights` argument in the `fit` function by including it in the training data.
#'
#' @return A scalar tensor representing the loss value.
#'
#' @examples
#' # Assuming you have tensors y_true and rv_y from a VGP model, and a kl_weight value
#' custom_loss <- loss(y_true, rv_y, kl_weight)
#'
loss <- function(y, rv_y, kl_weight = NULL, sample_weights = NULL, jitter = 1e-6) {
    variational_loss <- rv_y$variational_loss(y, kl_weight = kl_weight)

    # Apply jitter for numerical stability
    variational_loss <- variational_loss + jitter

    # Apply sample_weights if provided
    if (!is.null(sample_weights)) {
        weighted_loss <- variational_loss * sample_weights
        return(tf$reduce_mean(weighted_loss))
    }

    return(variational_loss)
}

loss <- tf_function(loss)

#' Create a TensorFlow dataset from a dataframe
#'
#' This function converts a given dataframe into a TensorFlow dataset that can be used
#' as input for a Keras model during training or evaluation.
#'
#' @param df A dataframe containing the input features and the target variable.
#'           The target variable should be in the last column.
#' @param batch_size An integer representing the number of samples per gradient update during training.
#' @param shuffle A boolean flag indicating whether to shuffle the dataset before batching. Default is TRUE.
#' @param weights An optional vector of weights for each observation in the dataset.
#'
#' @return A TensorFlow dataset object that can be used for model training or evaluation.
#'
#' @examples
#' # Create a TensorFlow dataset from a sample dataframe
#' sample_df <- data.frame(x1 = rnorm(100), x2 = rnorm(100), y = rnorm(100))
#' dataset <- create_dataset(sample_df, batch_size = 32)
#'
#' @references
#' The structure for the `tensor_slices_dataset()` function when passing sample weights was taken from:
#' https://stackoverflow.com/questions/66802860/how-to-use-sample-weights-with-tensorflow-datasets
#'
create_dataset <- function(df, batch_size, shuffle = FALSE, weights = NULL) {
    # Convert dataframe to matrix
    df <- as.matrix(df)

    if (!is.null(weights)) {
        ds <- tensor_slices_dataset(list(df[, 1:(ncol(df) - 1)], df[, ncol(df), drop = FALSE], weights))
    } else {
        # Create a TensorFlow dataset from the matrix, splitting input features and target variable
        ds <- tensor_slices_dataset(list(df[, 1:(ncol(df) - 1)], df[, ncol(df), drop = FALSE]))
    }
    # Shuffle the dataset if specified
    if (shuffle) {
        ds <- ds %>% dataset_shuffle(buffer_size = nrow(df), seed = 42L)
    }

    # Batch the dataset with the specified batch_size
    ds %>% dataset_batch(batch_size = batch_size)
}

#' Gaussian Process Regression using Variational Gaussian Process (VGP) layer
#'
#' This function trains a Gaussian Process Regression model using a VGP layer in a Keras model.
#' It takes in training and test data, as well as other hyperparameters and options for the model,
#' and returns the trained model, predictions, and training history.
#'
#' @param train_data A dataframe containing the training data, with the target variable in the last column.
#' @param test_data A dataframe containing the test data, with the target variable in the last column.
#' @param num_epochs An integer representing the number of times the model is trained over the entire dataset. Default is 200.
#' @param batch_size An integer representing the number of samples per gradient update during training. Default is 0.5% of the training observations.
#' @param defined_lr A scalar representing the learning rate for the optimizer. Default is 0.008.
#' @param num_inducing_points An integer representing the number of inducing points for the VGP layer. Default is 1% of the training observations.
#' @param n_tfd_samples An integer representing the number of samples to draw from the VGP layer for prediction. Default is 1000.
#' @param weights_vector An optional tensor of the same shape as the target variable `y` in `train_data`, representing the weights of each observation. Default is NULL.
#' @param keras_weighted_mode A boolean flag indicating whether to use the Keras method of applying sample weights (i.e., similar to using the `sample_weight` argument in the `fit` function). Default is TRUE.
#' @param patience An integer representing the number of epochs with no improvement after which training will be stopped. Default is 10% of the number of epochs.
#'
#' @return A list containing the trained model (`model`), predictions for the test data (`predictions`), and the training history (`history`).
#'
#' @examples
#' # Assuming you have dataframes train_data and test_data with input features and the target variable
#' result <- gpr(train_data, test_data)
#' trained_model <- result$model
#' predictions <- result$predictions
#' training_history <- result$history
#'
gpr <- function(train_data, test_data, num_epochs = 200, defined_lr = 0.008, num_inducing_points = NULL, batch_size = NULL, n_tfd_samples = 1000, weights_vector = NULL, keras_weighted_mode = TRUE, early_stopping_patience = NULL, jitter = 1e-6) {
    # Set default values for batch_size, num_inducing_points, and early_stopping_patience
    if (is.null(batch_size)) {
        batch_size <- ceiling(nrow(train_data) * 0.01) # 0.005
    }
    if (is.null(num_inducing_points)) {
        num_inducing_points <- ceiling(nrow(train_data) * 0.01)
    }
    if (is.null(early_stopping_patience)) {
        early_stopping_patience <- ceiling(num_epochs * 0.10)
    }

    # Create datasets (based on whether the weights were provided or not)
    if (!is.null(weights_vector) && keras_weighted_mode) {
        train_ds <- create_dataset(train_data, batch_size = batch_size, weights = weights_vector)
    } else {
        train_ds <- create_dataset(train_data, batch_size = batch_size)
    }

    train_ds <- train_ds %>% dataset_cache()

    test_ds <- create_dataset(test_data, batch_size = nrow(test_data), shuffle = FALSE)

    # Not sure, but could be also useful here
    # keras::use_session_with_seed(42)
    tensorflow::set_random_seed(42)

    # Define the model
    model <- keras_model_sequential() %>%
        layer_dense(units = ncol(train_data) - 1, input_shape = ncol(train_data) - 1, use_bias = FALSE) %>%
        # layer_variational_gaussian_process(
        #     num_inducing_points = num_inducing_points,
        #     kernel_provider = RBFKernelFn(),
        #     # kernel_provider = MaternKernelFn(),
        #     # kernel_provider =  MaternFiveHalvesKernelFn(),
        #     event_shape = 1,
        #     inducing_index_points_initializer = initializer_constant(as.matrix(train_data[sample(seq_len(nrow(train_data)), num_inducing_points), 1:(ncol(train_data) - 1)])),
        #     unconstrained_observation_noise_variance_initializer = initializer_constant(array(0.1))
        # )
        layer_variational_gaussian_process(
            num_inducing_points = num_inducing_points,
            kernel_provider = RBFKernelFn(),
            event_shape = 1,
            inducing_index_points_initializer = initializer_constant(as.matrix(train_data[sample(seq_len(nrow(train_data)), num_inducing_points), 1:(ncol(train_data) - 1)])),
            unconstrained_observation_noise_variance_initializer = initializer_constant(array(0.1))
        )


    # KL weight sums to one for one epoch
    kl_weight <- batch_size / nrow(train_data)
    # kl_weight <- 0.9

    desired_loss <- if (!is.null(weights_vector) && !keras_weighted_mode) {
        purrr::partial(loss, kl_weight = kl_weight, sample_weights = weights_vector, jitter = jitter)
    } else {
        purrr::partial(loss, kl_weight = kl_weight, jitter = jitter)
    }

    model %>% compile(
        optimizer = optimizer_rmsprop(learning_rate = defined_lr),
        loss = desired_loss,
        metrics = list("mse", r2_metric)
    )

    # Add early stopping callback
    early_stopping_callback <- callback_early_stopping(
        monitor = "val_loss",
        patience = early_stopping_patience,
        restore_best_weights = TRUE
    )

    # Train the model
    history <- model %>% keras::fit(
        train_ds,
        epochs = num_epochs,
        validation_data = test_ds,
        verbose = 1,
        callbacks = list(early_stopping_callback) # Add the callback to the training
    )

    tensorflow::set_random_seed(42)
    # Make predictions with the new model
    yhats <- model(tf$convert_to_tensor(as.matrix(test_data[, 1:(ncol(test_data) - 1)])))
    yhat_samples <- yhats %>%
        tfd_sample(n_tfd_samples) %>%
        tf$squeeze() %>%
        tf$transpose()
    sample_means <- yhat_samples %>% apply(1, mean)

    return(list(model = model, distributions = yhats, predictions = sample_means, history = history))
}
