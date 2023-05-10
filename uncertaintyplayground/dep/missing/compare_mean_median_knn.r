# Mean imputation
mean_impute <- function(x, data) {
  missing_idx <- which(is.na(x))
  means <- colMeans(data, na.rm = TRUE)
  x[missing_idx] <- means[missing_idx]
  return(x)
}

# Median imputation
median_impute <- function(x, data) {
  missing_idx <- which(is.na(x))
  medians <- apply(data, 2, function(col) median(col, na.rm = TRUE))
  x[missing_idx] <- medians[missing_idx]
  return(x)
}

# k-Nearest Neighbors imputation
knn_impute <- function(x, data, k = 5) {
  distances <- apply(data, 1, function(row) euclidean_distance(x, row))
  nearest_indices <- order(distances)[1:k]
  nearest_neighbors <- data[nearest_indices, ]
  missing_idx <- which(is.na(x))
  
  for (idx in missing_idx) {
    x[idx] <- mean(nearest_neighbors[, idx], na.rm = TRUE)
  }
  
  return(x)
}

# Compute mean squared error (MSE)
mse <- function(x, y) {
  return(mean((x - y)^2))
}

# Generate example input with missing x1 value
input <- c(NA, 0.5, 0.5, 0.5)

# Get the true values for the missing attribute
true_values <- iris[, 1]

# Apply imputation methods
imputed_input_weighted <- weighted_impute(input, iris_norm[, 1:4], probabilities, min_values, max_values)
imputed_input_mean <- mean_impute(input, iris[, 1:4])
imputed_input_median <- median_impute(input, iris[, 1:4])
imputed_input_knn <- knn_impute(input, iris_norm[, 1:4], k = 5)

# Compute the MSE for each method
mse_weighted <- mse(imputed_input_weighted[1], true_values)
mse_mean <- mse(imputed_input_mean[1], true_values)
mse_median <- mse(imputed_input_median[1], true_values)
mse_knn <- mse(round(imputed_input_knn[1],2), true_values)

# Print the MSE for each method
cat("MSE - Weighted Imputation:", mse_weighted, "\n")
cat("MSE - Mean Imputation:", mse_mean, "\n")
cat("MSE - Median Imputation:", mse_median, "\n")
cat("MSE - k-Nearest Neighbors Imputation:", mse_knn, "\n")
