normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

iris_norm <- iris
iris_norm[, 1:4] <- lapply(iris[, 1:4], normalize)

euclidean_distance <- function(a, b) {
  return (sqrt(sum((a[!is.na(a) & !is.na(b)] - b[!is.na(a) & !is.na(b)])^2, na.rm = TRUE)))
}

softmax <- function(x) {
  exp_x <- exp(x)
  return (exp_x / sum(exp_x))
}

distance_based_prob <- function(x, data) {
  distances <- apply(data, 1, function(row) euclidean_distance(x, row))
  inverted_distances <- 1 / distances
  softmax(inverted_distances)
}

weighted_impute <- function(x, data, probabilities, min_values, max_values) {
  missing_idx <- which(is.na(x))
  
  for (idx in missing_idx) {
    weighted_value <- sum(data[, idx] * probabilities, na.rm = TRUE)
    x[idx] <- weighted_value
  }
  
  # Convert imputed values back to their original scales
  x <- (x * (max_values - min_values)) + min_values
  
  return(x)
}

# Get the min and max values of each feature in the original dataset
min_values <- sapply(iris[, 1:4], min)
max_values <- sapply(iris[, 1:4], max)

# Example input with missing x1 value
input <- c(NA, 0.5, 0.5, 0.5)

# Calculate probabilities
probabilities <- distance_based_prob(input, iris_norm[, 1:4])

# Impute missing value(s) using softmax-weighted values and convert them back to their original scales
imputed_input <- weighted_impute(input, iris_norm[, 1:4], probabilities, min_values, max_values)

# Print imputed input
print(imputed_input)

# This method ca be described as a variation of the k-Nearest Neighbors (k-NN) imputation method with softmax-weighted distances. The standard k-NN imputation method uses the mean or median of the k-nearest neighbors' values to impute missing data. In this variation, we're using the softmax function to convert the distances between the input and instances into a probability distribution, and then using these probabilities as weights to compute a weighted average for imputing missing values.