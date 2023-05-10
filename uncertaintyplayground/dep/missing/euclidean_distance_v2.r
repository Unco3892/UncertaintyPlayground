# Imagine I would like to predict which instance of a dataset most likely relates to a given (single) variable input by applying a softmax to all the instances. The idea here is not predicting a class (y) but rather which instance it was. So at the end you get a probability distribution for all the instances of a given variable at the inference. The idea is that I have two variables x1 and x2 which are all available (non-missing). However, at the time of inference, x1 or x2 may be missing, therefore, I would like to find the probability of all instances.

# What is this called? Also, implement this in R in form of a function so that it can be applied to any dataset.

# Load required libraries
library(dplyr)
library(tidyr)

# Define distance function
euclidean_distance <- function(x1, x2) {
  sqrt(sum((x1 - x2)^2))
}

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

# Modified instance_prediction function
# instance_prediction <- function(data, input_variables) {
#   n <- nrow(data)
  
#   if (n == 0) {
#     stop("Data frame must have at least one row.")
#   }
  
#   # Calculate distances using Euclidean distance
#   distances <- sqrt(rowSums((data - input_variables) ^ 2, na.rm = TRUE))
  
#   # Handle cases where the dataset has only one row
#   if (n == 1) {
#     return(data.frame(Instance = rownames(data), Probability = 1))
#   }
  
#   # Compute softmax probabilities
#   exp_neg_distances <- exp(-distances)
#   probabilities <- exp_neg_distances / sum(exp_neg_distances)
  
#   return(data.frame(Instance = rownames(data), Probability = probabilities))
# }

# Test the function on a sample dataset
data <- data.frame(x1 = c(1, 2, 3, 4, 5), x2 = c(5, 4, 3, 2, 1), x3 = c(6, 7, 8, 9, 10))
input_variables <- c(2.5, NA, 8)
result <- instance_prediction(data, input_variables)
print(result)

# Therefore, use some principles of object-orientated programming 
# did not normalize the data

#----------------------------------------------------------------------------------------
#Here are two approaches to impute the missing value using the instance probabilities:

# Weighted average: Compute the weighted average of the missing variable using the calculated instance probabilities as weights.
# Highest probability instance: Select the instance with the highest probability and use its value for the missing variable.

# Impute missing value using weighted average
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

# Impute missing value using the highest probability instance
impute_highest_probability_instance <- function(data, input_variables, result) {
  missing_indices <- which(is.na(input_variables))
  
  if (length(missing_indices) == 0) {
    return(input_variables)
  }
  
  highest_probability_instance <- which.max(result$Probability)
  imputed_values <- input_variables
  imputed_values[missing_indices] <- data[highest_probability_instance, missing_indices]
  
  return(imputed_values)
}

# Test imputation methods
data <- data.frame(x1 = c(1, 2, 3, 4, 5), x2 = c(5, 4, 3, 2, 1), x3 = c(6, 7, 8, 9, 10))
input_variables <- c(2.5, NA, 8)
result <- instance_prediction(data, input_variables)

# Weighted average imputation (without k)
imputed_weighted_average <- impute_weighted_average_k(data, input_variables, result)
print(imputed_weighted_average)

# Highest probability instance imputation
imputed_highest_probability_instance <- impute_highest_probability_instance(data, input_variables, result)
print(imputed_highest_probability_instance)
