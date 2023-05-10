# Load libraries
library(kernlab)
library(ggplot2)

# Create some sample data
X <- matrix(c(0, 1, 1, 0, 1, 1, 2, 2), nrow=4, ncol=2, byrow=TRUE)

# Calculate the RBF kernel matrix
sigma <- 1.0  # This is the parameter for the RBF kernel, adjust as needed
rbf_kernel <- rbfdot(sigma=sigma)
K <- kernelMatrix(rbf_kernel, X)

# Select specific instances (e.g., the first and the third instances)
instance_indices <- c(1, 3)
instance_similarities <- K[instance_indices,]

# Convert the similarity matrix to a data frame for plotting
plot_data <- data.frame(
  Instance = factor(rep(instance_indices, each = nrow(X))),
  SimilarTo = factor(rep(1:nrow(X), length(instance_indices))),
  Similarity = c(instance_similarities)
)

# Create the similarity plot using ggplot2
ggplot(plot_data, aes(x = SimilarTo, y = Similarity, fill = Instance)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Similarity Scores for Selected Instances",
       x = "Instances",
       y = "Similarity Score") +
  theme_minimal() +
  scale_fill_discrete(name = "Selected\nInstance")
