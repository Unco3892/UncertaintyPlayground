# Load required libraries
library(ggplot2) # for the diamonds dataset
library(dplyr)
library(doParallel)
library(foreach)

source("scripts/missing/euclidean_distance_v2.r")

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

seed_parallel <- function(seed) {
  set.seed(seed)
  seed
}

# Function to calculate the error for a single test case
calc_error <- function(data, test_case, true_value, missing_index) {
  # Apply instance-based imputation
  result <- instance_prediction(data, test_case)
  imputed_instance_based <- impute_highest_probability_instance(data, test_case, result)
  error_instance_based <- abs(true_value - imputed_instance_based[missing_index])

  # Apply mean and median imputation
  imputed_mean_median <- impute_mean_median(data, test_case)
  error_mean <- abs(true_value - imputed_mean_median[missing_index])
  error_median <- abs(true_value - imputed_mean_median[missing_index + 1])

  return(c(error_instance_based, error_mean, error_median))
}

# Register parallel backend
n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Run test cases in parallel with a fixed seed
seed_base <- 1234
# Number of test cases
n_tests <- 1000

# Prepare the diamonds dataset
data <- diamonds %>% select(-c(cut, color, clarity))
data <- scale(data) # Normalize the dataset

# Prepare the iris dataset
# data <- iris %>% mutate(Species = as.factor(Species))
# data <- model.matrix(~ . + 0, data) # One-hot encoding for the categorical feature
# data <- as.data.frame(data)
# rownames(data) <- 1:nrow(data)
# data <- scale(data) # Normalize the dataset

# Run test cases in parallel
errors <- foreach(i = 1:n_tests, .combine = 'rbind', .packages = c("dplyr", "tidyr"), .options.multicore = list(set.seed = TRUE)) %dopar% {
  # Set seed for each task
  seed_parallel(seed_base + i)

  # Select a random sample with a missing value
  index <- sample(nrow(data), 1)
  test_case <- data[index, ]
  test_data <- data[-index, ]
  missing_index <- sample(ncol(test_data), 1)
  true_value <- test_case[missing_index]
  test_case[missing_index] <- NA

  calc_error(test_data, test_case, true_value, missing_index)
}

# Stop parallel backend
stopCluster(cl)

# Calculate the average error for each imputation method
error_instance_based <- mean(errors[, 1])
error_mean <- mean(errors[, 2])
error_median <- mean(errors[, 3])

cat("Instance-based imputation error:", error_instance_based, "\n")
cat("Mean imputation error:", error_mean, "\n")
cat("Median imputation error:", error_median, "\n")
