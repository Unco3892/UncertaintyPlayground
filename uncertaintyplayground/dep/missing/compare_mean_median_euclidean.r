# Load required packages
library(mice)  # For imputation methods

# Replace your_dataset with your dataset's name
data <- iris

# Set the percentage of missing values
missing_percentage <- 0.2

# Introduce missing values randomly
set.seed(42)
data_na <- data
numeric_columns <- which(sapply(data_na, is.numeric))
for (i in numeric_columns) {
  data_na[sample(1:nrow(data_na), size = floor(nrow(data_na) * missing_percentage)), i] <- NA
}

# Function to calculate RMSE
rmse <- function(actual, predicted) {
  return(sqrt(mean((actual - predicted)^2, na.rm = TRUE)))
}

# Calculate RMSE for each imputation method
# rmse_mean <- sapply(numeric_columns, function(i) rmse(data[, i], mean_imputed_data[, i]))
# rmse_median <- sapply(numeric_columns, function(i) rmse(data[, i], median_imputed_data[, i]))
# rmse_softmax_weighted <- sapply(numeric_columns, function(i) rmse(data[, i], softmax_weighted_imputed_data[, i]))

# Compare RMSE values
# cat("RMSE for mean imputation:\n")
# print(rmse_mean)
# cat("RMSE for median imputation:\n")
# print(rmse_median)
# cat("RMSE for softmax-weighted imputation:\n")
# print(rmse_softmax_weighted)

# Calculate performance metrics for each imputation method
perf_mean <- sapply(numeric_columns, function(i) caret::postResample(pred = mean_imputed_data[, i], obs = data[, i]))
perf_median <- sapply(numeric_columns, function(i) caret::postResample(pred = median_imputed_data[, i], obs = data[, i]))
perf_softmax_weighted <- sapply(numeric_columns, function(i) caret::postResample(pred = softmax_weighted_imputed_data[, i], obs = data[, i]))

# Compare performance metrics
cat("Performance metrics for mean imputation:\n")
print(perf_mean)
cat("Performance metrics for median imputation:\n")
print(perf_median)
cat("Performance metrics for softmax-weighted imputation:\n")
print(perf_softmax_weighted)

library(missForest)
miss_forest_imputed_data <- missForest(data_na, xtrue = data)
