z_R_flat <- matrix(nrow = train_smp_size * R, ncol = hidden)
for (l in 1:hidden) {
  z_R_flat[, l] <- as.vector(z_R[, l, ])
}
z_R_cov <- cov(z_R_flat)

# Number of test points
n_test <- nrow(x_test)

# Initialize vectors to store the predicted outcomes and confidence intervals
y_pred <- numeric(n_test)
y_lower <- numeric(n_test)
y_upper <- numeric(n_test)

n_simulations = 1000
alpha = 0.95

i <- 1
# Calculate predictions and confidence intervals for each test point
for (i in 1:n_test) {
  x_new <- t(x_test[i, ])
  Z_new <- predict(modL_p, data.frame(x_new))
  Z_samples <- MASS::mvrnorm(n = n_simulations, mu = Z_new, Sigma = z_R_cov)
  y_samples <- predict(modR_p, data.frame(Z_samples))
  y_quantiles <- quantile(y_samples, probs = c((1 - alpha) / 2, (1 + alpha) / 2))
  
  y_pred[i] <- mean(y_samples)
  y_lower[i] <- y_quantiles[1]
  y_upper[i] <- y_quantiles[2]
}

# Store the results in a data frame
results <- data.frame(y_true = y_test, y_pred = y_pred, y_lower = y_lower, y_upper = y_upper)

# Add a new column to the results data frame for the legend
results$Type <- "Predicted"

# Create a new data frame for the ground truth values
true_results <- data.frame(y_true = y_test, y_pred = y_test, y_lower = NA, y_upper = NA, Type = "True")

# Combine the results and true_results data frames
plot_data <- rbind(results, true_results)

# Plot the results
library(ggplot2)

ggplot(plot_data, aes(x = y_true, y = y_pred, color = Type)) +
  geom_point(alpha = 0.5, size = 3) +
  geom_errorbar(aes(ymin = y_lower, ymax = y_upper), width = 0.2, color = "red", alpha = 0.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", size = 1.2) +
  labs(
    title = "True Outcomes vs Predicted Outcomes with Confidence Intervals",
    x = "True Outcome",
    y = "Predicted Outcome"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 14),
    axis.title = element_text(size = 16, face = "bold"),
    axis.text = element_text(size = 12),
    title = element_text(size = 18, face = "bold"),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_line(color = "gray95")
  ) +
  scale_color_manual(
    name = "Type",
    values = c("Predicted" = "blue", "True" = "darkorange"),
    labels = c("Predicted Value", "True Value")
  )


# Number of test points
n_test <- nrow(x_test)

# Calculate the Z_new values for all test points at once
Z_new <- predict(modL_p, x_test)

# Define a function to compute the predictions and confidence intervals for each Z_new
compute_y_pred_and_ci <- function(z_new) {
  Z_samples <- MASS::mvrnorm(n = n_simulations, mu = z_new, Sigma = z_R_cov)
  y_samples <- predict(modR_p, data.frame(Z_samples))
  y_quantiles <- quantile(y_samples, probs = c((1 - alpha) / 2, (1 + alpha) / 2))

  return(c(mean(y_samples), y_quantiles))
}

# Apply the function to each row of Z_new
predictions_and_cis <- t(apply(Z_new, 1, compute_y_pred_and_ci))

# Store the results in vectors
y_pred <- predictions_and_cis[, 1]
y_lower <- predictions_and_cis[, 2]
y_upper <- predictions_and_cis[, 3]
