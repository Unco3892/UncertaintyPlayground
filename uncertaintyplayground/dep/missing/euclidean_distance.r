# Imagine I would like to predict which instance of a dataset most likely relates to a given (single) variable input by applying a softmax to all the instances. The idea here is not predicting a class (y) but rather which instance it was. So at the end you get a probability distribution for all the instances of a given variable at the inference. The idea is that I have two variables x1 and x2 which are all available (non-missing). However, at the time of inference, x1 or x2 may be missing, therefore, I would like to find the probability of all instances.

# What is this called? And how do I implement it in R with iris data?

library(tidyverse)

data(iris)

softmax <- function(x) {
  exp_x <- exp(x)
  return(exp_x / sum(exp_x))
}

euclidean_distance <- function(a, b) {
  common_features <- intersect(names(a)[!is.na(a)], names(b)[!is.na(b)])
  sum((a[common_features] - b[common_features])^2)
}
softmax <- function(x) {
  exp_x <- exp(x)
  return (exp_x / sum(exp_x))
}

instance_probabilities <- function(input, dataset) {
  distances <- apply(dataset, 1, function(row) euclidean_distance(input, row))
  softmax(-distances)  # Apply softmax to negative distances to get probabilities
}

# Sample input with x1 and x2 values (using Sepal.Length and Sepal.Width as an example)
input <- c(Sepal.Length = 1, Sepal.Width = NA)

# Compute instance probabilities
probabilities <- instance_probabilities(input, iris[, 1:2])

# Print probabilities
print(probabilities)

df <- tibble(instance = 1:nrow(iris), probability = probabilities)
ggplot(df, aes(x = instance, y = probability)) +
  geom_bar(stat = "identity") +
  labs(title = "Probability Distribution Over Instances",
       x = "Instance Index",
       y = "Probability") +
  theme_minimal()

which.max(probabilities)
iris[14,]
mean(iris$Sepal.Width)
iris[top5_indices,]$Sepal.Width
# Get the indices that sort the vector in descending order
sorted_indices <- order(probabilities, decreasing = TRUE)

# Get the top 5 values and their indices
top_5_values <- probabilities[sorted_indices[1:5]]
top_5_indices <- sorted_indices[1:5]

# Print the top 5 values and their indices
print(top_5_values)
print(top_5_indices)


mean(iris[top5_indices,]$Sepal.Width)
