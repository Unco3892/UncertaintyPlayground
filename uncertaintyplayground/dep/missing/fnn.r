# Isn't what Marc-O proposed simply nearest neighbour?
install.packages("FNN")
library(FNN)

# Example data
data <- data.frame(x1 = c(1, 2, 3, 4, 5), x2 = c(5, 4, 3, 2, 1))

find_nearest_neighbor <- function(my_query, data) {
  # If x1 is missing, use x2 to find the nearest neighbor
  if (is.na(my_query$x1)) {
    data_x2 <- data.frame(x2 = data$x2)
    nearest_neighbor <- get.knnx(data= data_x2, query = matrix(my_query$x2), k = 1)
    missing_value <- data[nearest_neighbor$nn.index, "x1"]
  }
  # If x2 is missing, use x1 to find the nearest neighbor
  else if (is.na(my_query$x2)) {
    data_x1 <- data.frame(x1 = data$x1)
    # nearest_neighbor <- get.knn(data_x1, matrix(query$x1, nrow = 1), k = 1)
    nearest_neighbor <- get.knnx(data= data_x1, query = matrix(my_query$x1), k = 1)
    missing_value <- data[nearest_neighbor$nn.index, "x2"]
  }
  return(missing_value)
}

# For missing x1
data_x2 <- data.frame(x2 = data$x2)

# For missing x2
data_x1 <- data.frame(x1 = data$x1)

# Example query with missing x1
query1 <- data.frame(x1 = NA, x2 = 2.5)
predicted_x1 <- find_nearest_neighbor(query1, data)
print(predicted_x1)

# Example query with missing x2
query2 <- data.frame(x1 = 2.5, x2 = NA)
predicted_x2 <- find_nearest_neighbor(query2, data)
print(predicted_x2)


# Extract the feature vectors for the iris dataset
iris_features <- iris[, 1:4]

# Define the input variable as the first instance in the iris dataset
input_variable <- iris_features[1, ]

# Define the dataset instances as the remaining instances in the iris dataset
dataset_instances <- iris_features[-1, ]