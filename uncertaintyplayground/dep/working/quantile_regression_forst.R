# Load required packages
library(quantregForest)

# split <- initial_split(dummy_data, prop = 0.05)
split <- initial_split(dummy_data, prop = 0.2)
train_main <- training(split)
test_main <- testing(split)

train_sample_weights <- pull(train_main, w_comp)
train <- select(train_main, -w_comp)
test <- select(test_main, -w_comp)

set.seed(1)
# Fit a quantile random forest to the training data
qrf <- quantregForest(dplyr::select(train, - y_comp), train$y_comp, weights = train_main$w_comp) #, ntree = 500 , nthreads = 5

# Make predictions on the test data and compute quantiles
pred <- predict(qrf, newdata = dplyr::select(test, - y_comp), type = "quantile", p = c(0.1, 0.5, 0.9))

# Plot the predicted values and quantiles
df <- data.frame(
  actual = test$y_comp,
  pred = pred[,2],
  lower = pred[,1],
  upper = pred[,3]
)

caret::postResample(pred = df$pred, obs = df$actual)

ggplot(df, aes(x = actual, y = pred)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, fill = "red") +
  xlab("Actual Value") +
  ylab("Predicted Value")


