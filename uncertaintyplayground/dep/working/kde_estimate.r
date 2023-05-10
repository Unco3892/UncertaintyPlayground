library(ggplot2)

# Generate synthetic data
set.seed(42)
X <- runif(100, 0, 10)
y <- 2 * X + 1 + rnorm(100, sd = 2)

# Plot the data
data <- data.frame(X = X, y = y)
ggplot(data, aes(x = X, y = y)) + geom_point() + theme_minimal()


# Fit a linear regression model
model <- lm(y ~ X, data = data)

# Make predictions
data$y_pred <- predict(model, data)

# Plot the predictions
ggplot(data, aes(x = X)) +
  geom_point(aes(y = y), color = "blue") +
  geom_line(aes(y = y_pred), color = "red") +
  theme_minimal()

# Calculate residuals
data$residuals <- residuals(model)

# Plot the KDE of residuals
ggplot(data, aes(x = residuals)) + 
  geom_density(fill = "blue", alpha = 0.5) + 
  theme_minimal()

# Choose a new observation
X_new <- 5

# Predict the outcome for the new observation
y_new_pred <- predict(model, newdata = data.frame(X = X_new))

# Calculate the 95% confidence interval
alpha <- 0.95
residual_quantiles <- quantile(data$residuals, c((1 - alpha) / 2, (1 + alpha) / 2))
y_new_conf_interval <- y_new_pred + residual_quantiles

cat("Predicted value:", y_new_pred, "\n")
cat(paste0(alpha * 100, "% confidence interval: (", y_new_conf_interval[1], ", ", y_new_conf_interval[2], ")\n"))

# Create a grid of X values
X_grid <- seq(min(X), max(X), length.out = 500)

# Calculate the confidence intervals for each X value
y_conf_intervals <- sapply(X_grid, function(x) {
  y_pred <- predict(model, newdata = data.frame(X = x))
  y_conf_interval <- y_pred + residual_quantiles
  return(y_conf_interval)
})

# Plot the confidence intervals
plot_data <- data.frame(
  X_grid = X_grid, 
  y_pred = predict(model, newdata = data.frame(X = X_grid)),
  y_lower = y_conf_intervals[1, ],
  y_upper = y_conf_intervals[2, ]
)

ggplot() +
  geom_point(data = data, aes(x = X, y = y), color = "blue", alpha = 0.5) +
  geom_line(data = plot_data, aes(x = X_grid, y = y_pred), color = "red") +
  geom_ribbon(data = plot_data, aes(x = X_grid, ymin = y_lower, ymax = y_upper), fill = "gray", alpha = 0.3) +
  theme_minimal()
