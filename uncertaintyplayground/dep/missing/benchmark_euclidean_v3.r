# Imagine I would like to predict which instance of a dataset most likely relates to a given (single) variable input by applying a softmax to all the instances. The idea here is not predicting a class (y) but rather which instance it was. So at the end you get a probability distribution for all the instances of a given variable at the inference. The idea is that I have two variables x1 and x2 which are all available (non-missing). However, at the time of inference, x1 or x2 may be missing, therefore, I would like to find the probability of all instances.

# What is this called? Also, implement this in R in form of a function so that it can be applied to any dataset.

# Load required libraries
library(dplyr)
library(tidyr)
library(tidyverse)

# Define distance function
euclidean_distance <- function(x1, x2) {
  sqrt(sum((x1 - x2)^2))
}

# for the VAE to work
if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()

library(keras)
K <- keras::backend()

# Define softmax function
softmax <- function(x) {
  exp_x <- exp(x)
  exp_x / sum(exp_x)
}

# Define instance-based prediction function
instance_prediction <- function(data, input_variables) {
  if (length(input_variables) != ncol(data)) {
    stop("The number of input variables must match the number of columns in the dataset.")
  }
  
  # Check for missing values
  missing_indices <- which(is.na(input_variables))

  # Calculate distances
  if (length(missing_indices) == length(input_variables)) {
    stop("All input variables cannot be missing.")
  } else {
    # Remove missing values from input_variables and dataset
    input_variables_filtered <- input_variables[-missing_indices]
    data_filtered <- data[ , -missing_indices, drop=FALSE]
    
    distances <- apply(data_filtered, 1, function(row) euclidean_distance(row, input_variables_filtered))
  }

  # Calculate probabilities using softmax
  probabilities <- softmax(-distances)

  # Return instance probabilities
  tibble(Instance = 1:nrow(data), Probability = probabilities)
}

# Impute missing value using the k-highest probability instances
impute_weighted_average_k <- function(data, input_variables, result, k = nrow(data)) {
  missing_indices <- which(is.na(input_variables))
  
  if (length(missing_indices) == 0) {
    return(input_variables)
  }
  
  imputed_values <- input_variables
  for (missing_index in missing_indices) {
    # Select top k instances
    top_k_instances <- head(result[order(-result$Probability), ], k)
    
    # Calculate weighted average using top k instances
    weighted_average <- sum(data[top_k_instances$Instance, missing_index] * top_k_instances$Probability)
    imputed_values[missing_index] <- weighted_average / sum(top_k_instances$Probability)
  }
  
  return(imputed_values)
}


# Load required libraries
library(ggplot2) # for the diamonds dataset
library(dplyr)
library(scales)
library(doParallel)
library(foreach)
# this library is for the random forest based imputation
library(missForest)
# this library is for the Bayesian imputation
library(Amelia)
# for the autoencoder
# turn off the warnings, especially the ones from the keras package
library(keras)
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 2)
library(tensorflow)
library(softImpute)
library(VIM)
# for the euclidean disance, we can alternatively use `proxy::dist`
library(proxy)
# for parallelization with progress bar
library(doSNOW)

impute_knn <- function(my_data, input_variables, k = 1) {
  missing_indices <- which(is.na(input_variables))
  
  if (length(missing_indices) == 0) {
    return(input_variables)
  }
  
  data_with_missing <- rbind(input_variables, my_data)
  imputed_data <- VIM::kNN(data_with_missing, k = k)
  imputed_values <- imputed_data[1, ]
  
  return(imputed_values)
}

impute_matrix_factorization <- function(my_data, input_variables) {
  missing_indices <- which(is.na(input_variables))
  
  if (length(missing_indices) == 0) {
    return(input_variables)
  }
  
  data_with_missing <- rbind(input_variables, my_data)
  complete_data <- softImpute::softImpute(data_with_missing)$u %*% diag(softImpute::softImpute(data_with_missing)$d) %*% t(softImpute::softImpute(data_with_missing)$v)
  imputed_values <- complete_data[1, ]
  
  return(imputed_values)
}

# Function to apply mean and median imputation
impute_mean_median <- function(data, input_variables) {
  missing_indices <- which(is.na(input_variables))
  
  if (length(missing_indices) == 0) {
    return(input_variables)
  }
  
  imputed_values <- input_variables
  for (missing_index in missing_indices) {
    imputed_values[missing_index] <- mean(data[, missing_index], na.rm = TRUE)
    imputed_values[missing_index + length(missing_indices)] <- median(data[, missing_index], na.rm = TRUE)
  }
  
  return(imputed_values)
}

impute_bayesian_amelia <- function(data, input_variables) {
  missing_indices <- which(is.na(input_variables))
  
  if (length(missing_indices) == 0) {
    return(input_variables)
  }
  
  data_with_missing <- rbind(input_variables, data)
  imputed_data <- amelia(data_with_missing, m = 1, ts = NULL, p2s = FALSE, parallel = "no")$imputations[[1]]
  imputed_values <- imputed_data[1, ]
  
  return(imputed_values)
}

seed_parallel <- function(seed) {
  set.seed(seed)
  seed
}

# I think because I did sigmoid, I'm learning "exactly" the input, right?

# defin the auto-encoder model
build_autoencoder <- function(data) {
  tensorflow::set_random_seed(42)

  input_dim <- ncol(data)

  # Encoder
  encoder_input <- layer_input(shape = input_dim)
  encoded <- encoder_input %>%
    # layer_dense(units = 64, activation = "linear") %>%
    # layer_dense(units = 32, activation = "relu")
    layer_dense(units = 20, activation = "LeakyReLU", kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001))

  # Decoder
  decoded <- encoded %>%
    layer_dense(units = input_dim, activation = "sigmoid") # Change the activation function to "linear"

  # Autoencoder
  autoencoder <- keras_model(inputs = encoder_input, outputs = decoded)

  # learning_rate = 0.1
  # Compile the model
  autoencoder %>% compile(optimizer ="adam", loss = "mean_absolute_error")  #mean_squared_error 
  #mean_absolute_error
  #mean_absolute_percentage_error

  return(autoencoder)
}

impute_autoencoder <- function(data, input_variables, autoencoder) {
  missing_indices <- which(is.na(input_variables))
  
  if (length(missing_indices) == 0) {
    return(input_variables)
  }
  
  data_with_missing <- rbind(input_variables, data)
  data_with_missing[is.na(data_with_missing)] <- 0
  
  imputed_data <- predict(autoencoder, data_with_missing, verbose = 0)
  imputed_values <- imputed_data[1, ]
  
  return(imputed_values)
}

# Define a function to build a simple variational autoencoder
build_vae <- function(input_shape, latent_dim = 10, hidden_dim = 5) {
  tensorflow::set_random_seed(42)
  input_shape <- ncol(input_shape)

  # Define the encoder model
  encoder_input <- layer_input(shape = input_shape)
  # encoder_hidden <- layer_dense(encoder_input, units = hidden_dim, activation = "LeakyReLU")
  encoder_hidden <- layer_dense(encoder_input, units = hidden_dim, activation = "LeakyReLU", kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001))

  z_mean <- layer_dense(encoder_hidden, units = latent_dim)
  z_log_var <- layer_dense(encoder_hidden, units = latent_dim)
  
  # Define a sampling function to generate latent vectors
  sampling <- function(args) {
    z_mean <- args[[1]]
    z_log_var <- args[[2]]
    epsilon <- k_random_normal(shape = k_shape(z_mean), mean = 0, stddev = 1)
    z <- z_mean + k_exp(z_log_var / 2) * epsilon
    z
  }
  
  # Use a lambda layer to apply the sampling function
  z <- layer_lambda(f = sampling)(list(z_mean, z_log_var))
  
  # Create the encoder model
  encoder <- keras_model(inputs = encoder_input, outputs = list(z, z_mean, z_log_var))
  
  # Define the decoder model
  decoder_input <- layer_input(shape = latent_dim)
  decoder_hidden <- layer_dense(decoder_input, units = hidden_dim, activation = "LeakyReLU")
  decoder_output <- layer_dense(decoder_hidden, units = input_shape, activation = "sigmoid")
  
  # Create the decoder model
  decoder <- keras_model(inputs = decoder_input, outputs = decoder_output)
  
  # Define the variational autoencoder model
  vae_output <- decoder(encoder(encoder_input)[[1]])
  vae <- keras_model(inputs = encoder_input, outputs = vae_output)
  
  # Define a custom loss function to include the KL divergence term
  # vae_loss <- function(x, x_decoded_mean){
  #   xent_loss <- (input_shape/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
  #   kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  #   xent_loss + kl_loss
  #   }
  # Define a custom loss function to include the KL divergence term
  vae_loss <- function(x, x_decoded_mean){
    mae_loss <- (input_shape/1.0)*loss_mean_absolute_error(x, x_decoded_mean)
    kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
    mae_loss + kl_loss
    }

  # Compile the variational autoencoder model
  vae %>% compile(optimizer = "adam", loss = vae_loss)
  
  # Return the encoder and the decoder models
  # list(encoder = encoder, decoder = decoder)
  return (vae)
}

# 4. Define a function to impute missing values using the VAE
impute_vae <- function(data, input_variables, vae) {
  missing_indices <- which(is.na(input_variables))
  
  if (length(missing_indices) == 0) {
    return(input_variables)
  }
  
  data_with_missing <- rbind(input_variables, data)
  data_with_missing[is.na(data_with_missing)] <- 0
  
  imputed_data <- predict(vae, data_with_missing, verbose = 0)
  imputed_values <- imputed_data[1, ]
  
  return(imputed_values)
}

# Function to calculate the error for a single test case
calc_error <- function(data, test_case, true_value, missing_index, autoencoder, vae, n_inst_missForest = 1000) {

  # Apply instance-based imputation
  message("Applying instance-based imputation...")
  result <- instance_prediction(data, test_case)
  imputed_instance_based <-  impute_weighted_average_k(data, test_case, result, k = 1)
  error_instance_based <- abs(true_value - imputed_instance_based[missing_index])

  # Apply weighted average imputation
  message("Applying weighted average imputation with all observations...")
  imputed_weighted_avg <-  impute_weighted_average_k(data, test_case, result)
  error_weighted_avg <- abs(true_value - imputed_weighted_avg[missing_index])

  # Apply weighted average imputation
  message("Applying weighted average imputation with 5 observations...")
  imputed_weighted_avg_5 <-  impute_weighted_average_k(data, test_case, result, k = 5)
  error_weighted_avg_5 <- abs(true_value - imputed_weighted_avg_5[missing_index])

  # Apply mean and median imputation
  message("Applying mean and median imputation...")
  imputed_mean_median <- impute_mean_median(data, test_case)
  error_mean <- abs(true_value - imputed_mean_median[missing_index])
  error_median <- abs(true_value - imputed_mean_median[missing_index + 1])

  # Apply missForest imputation
  message("Applying missForest...")
  # missForest cannot handle many missing instances so we have to sample some observations
  sampled_data <- data[sample(nrow(data), n_inst_missForest), ]
  data_with_missing <- rbind(test_case, sampled_data)
  imputed_missforest <- missForest(data_with_missing, ntree = 200)$ximp[1,] #`1` is the first row since our test_case is only one observation
  error_missforest <- abs(true_value - imputed_missforest[missing_index])

  # Apply Bayesian imputation using Amelia
  message("Applying Bayesian (Bootrapped EM) imputation...")
  imputed_bayesian <- impute_bayesian_amelia(data, test_case)
  error_bayesian <- abs(true_value - imputed_bayesian[missing_index])
  
  # Apply autoencoder imputation
  message("Applying autoencoder imputation...")
  imputed_autoencoder <- impute_autoencoder(data, test_case, autoencoder)
  error_autoencoder <- abs(true_value - imputed_autoencoder[missing_index])
  
  # Apply matrix factorization imputation
  message("Applying matrix factorization imputation...")
  imputed_matrix_factorization <- impute_matrix_factorization(data, test_case)
  error_matrix_factorization <- abs(true_value - imputed_matrix_factorization[missing_index])

  # Apply kNN imputation
  message("Applying kNN imputation...")
  imputed_knn <- impute_knn(data, test_case)
  error_knn <- as.numeric(abs(true_value - imputed_knn[missing_index]))

  # Apply VAE imputation
  message("Applying VAE imputation...")
  imputed_vae <- impute_vae(data, test_case, vae)
  error_vae <- abs(true_value - imputed_vae[missing_index])


  return(c(error_instance_based, error_weighted_avg, error_weighted_avg_5, error_mean, error_median, error_missforest, error_bayesian, error_autoencoder, error_matrix_factorization, error_knn,error_vae))
}

# Run test cases in parallel with a fixed seed
seed_base <- 1234
# Number of test cases
n_tests <- 200

# Prepare the diamonds dataset
data <- diamonds %>% select_if(is.numeric)
# data <- scale(data) # Normalize the dataset
# Get the original ranges
original_ranges <- apply(data, 2, range)
scaled_df <- mutate_all(data, scales::rescale)  # Min-Max scaling
# Define a function to unscale the data
unscale <- function(x, original_range) {
  x * (original_range[2] - original_range[1]) + original_range[1]
}
# Unscaled the scaled data
unscaled_df <- mutate_all(scaled_df, unscale, original_range = original_ranges)
data <- as.matrix(scaled_df)

# Prepare the iris dataset
# data <- iris %>% mutate(Species = as.factor(Species))
# data <- iris %>% select_if(is.numeric)
# data <- model.matrix(~ . + 0, data) # One-hot encoding for the categorical feature
# data <- as.data.frame(data)
# original_ranges <- apply(data, 2, range)
# scaled_df <- mutate_all(data, scales::rescale)  # Min-Max scaling
# unscaled_df <- mutate_all(scaled_df, unscale, original_range = original_ranges)
# data <- as.matrix(scaled_df)

# Split the data into training and test sets
library(caret)
set.seed(1)
split_indices <- createDataPartition(data[, 1], p = 0.5, list = FALSE)
df_train_data <- data[split_indices, ]
df_test_data <- data[-split_indices, ]

# build the autoencoder model (on the entire data)
early_stopping <- callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE)

# Train the autoencoder model
autoencoder <- build_autoencoder(df_train_data)
autoencoder %>% fit(df_train_data, df_train_data,
                     epochs = 1000,
                     batch_size = 64,
                     verbose = 1,
                     validation_split = 0.2,
                     callbacks = list(early_stopping))

# Save autoencoder model to a file  
keras::save_model_hdf5(autoencoder, "autoencoder.h5")

# Train the VAE model
vae_model <- build_vae(df_train_data)
vae_model %>% fit(df_train_data, df_train_data,
                  epochs = 1000,
                  batch_size = 64,
                  verbose = 1,
                  validation_split = 0.2,
                  callbacks = list(early_stopping))

# Save VAE model to a file
keras::save_model_weights_hdf5(vae_model, "vae.h5")

# Register parallel backend
n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
registerDoSNOW(cl)

pb <- txtProgressBar(max = n_tests, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

# Run test cases in parallel
errors <- foreach(i = 1:n_tests, .packages = c("dplyr", "tidyr", "missForest", "Amelia", "keras"), .options.multicore = list(set.seed = TRUE), .options.snow = opts) %dopar% {
  # Set seed for each task
  seed_parallel(seed_base + i)

  # Select a random sample with a missing value
  index <- sample(nrow(df_test_data), 1)
  test_case <- df_test_data[index, ]
  test_data <- df_test_data[-index, ]
  missing_index <- sample(ncol(test_data), 1)
  true_value <- test_case[missing_index]
  test_case[missing_index] <- NA
    
  # Load the autoencoder model from the file
  autoencoder <- keras::load_model_hdf5("autoencoder.h5")

  # Load the VAE model weights from the file
  vae_model <- build_vae(test_data)
  keras::load_model_weights_hdf5(vae_model, "vae.h5")

  # Report progress
  calc_error(test_data, test_case, true_value, missing_index, autoencoder,vae_model)
}
close(pb)
stopCluster(cl)

# Convert the output to a dataframe
errors_df <- data.frame(matrix(unlist(errors), nrow=length(errors), byrow=TRUE))
colnames(errors_df) <- paste0("approach_", seq_along(errors[[1]]))
errors_df$target_var <- map_chr(errors, ~ names(.x)[which(!is.na(.x))[1]])
# turn the errors into percentages
errors_df <- errors_df %>% 
  mutate_all(~ ifelse(is.numeric(.), . * 100, .))

# Calculate the average error for each imputation method
error_instance_based <- mean(errors_df[, 1])
error_weighted_avg <- mean(errors_df[, 2])
error_weighted_avg_5 <- mean(errors_df[, 3])
error_mean <- mean(errors_df[, 4])
error_median <- mean(errors_df[, 5])
error_missforest <- mean(errors_df[, 6])
error_bayesian <- mean(errors_df[, 7])
error_autoencoder <- mean(errors_df[, 8])
error_matrix_factorization <- mean(errors_df[, 9])
error_knn <- mean(errors_df[, 10])
error_vae <- mean(errors_df[, 11])

cat("Instance-based imputation error:", error_instance_based, "\n")
cat("Weighted average imputation error (all observations):", error_weighted_avg, "\n")
cat("Weighted average imputation error (5 observations):", error_weighted_avg_5, "\n")
cat("Mean imputation error:", error_mean, "\n")
cat("Median imputation error:", error_median, "\n")
cat("MissForest imputation error:", error_missforest, "\n")
cat("Bayesian imputation error:", error_bayesian, "\n")
cat("Autoencoder imputation error:", error_autoencoder, "\n")
cat("Matrix factorization imputation error:", error_matrix_factorization, "\n")
cat("kNN imputation error:", error_knn, "\n")
cat("VAE imputation error:", error_vae, "\n")

# Compare different imputation methods
comparison_df <- data.frame(
  Method = c("Instance-based", "Weighted average (all)", "Weighted average (5)", "Mean", "Median", "MissForest", "Bayesian", "Autoencoder", "Matrix factorization", "kNN", "VAE"),
  Error = c(error_instance_based, error_weighted_avg, error_weighted_avg_5, error_mean, error_median, error_missforest, error_bayesian, error_autoencoder, error_matrix_factorization, error_knn,error_vae)
)

# Sort the methods by their error
comparison_df <- comparison_df[order(comparison_df$Error), ]

# Plot a bar chart to visualize the comparison
ggplot(comparison_df, aes(x = reorder(Method, Error), y = Error, fill = c(Method))) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(x = "Imputation Method", y = "Mean Error (%)") +
  theme_minimal()+
  guides(fill = "none")


