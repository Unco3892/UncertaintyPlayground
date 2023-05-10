# Load the required libraries
library(keras)

# Load the efficientnet model
model <- application_efficientnet_b0(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(224, 224, 3)
)

# Freeze the layers except the last 3

freeze_weights(model, from = -1)

# Add a new classification layer
x <- layer_global_average_pooling_2d(model$output)
x %<>% layer_dense(units = 128, activation = "relu")
x %<>% layer_dropout(rate = 0.5)
x %<>% layer_dense(units = 1, activation = "linear")

# Create the final model
model_final <-
  keras_model(inputs = model$input, outputs = x)

# Compile the model
model_final %>% compile(
  loss = "mse",
  metrics = list(
    metric_root_mean_squared_error(name = "rmse"),
    tfaddons::metric_rsquare()
  ),
  optimizer = optimizer_adam(learning_rate = 0.001),
  # optimizer = optimizer_adamax(0.002) #lr_schedule
)

# Define the learning rate schedule
# lr_schedule <- function(epoch, lr) {
#   return (0.001 * exp(-0.1 * epoch))
# }

# Fit the model to the training data
hist <-
  model_final %>%
  fit(
    x = x_train_img,
    y = y_train,
    epochs = 100,
    batch_size = 32,
    callbacks = list(
      callback_early_stopping(
        patience = 15,
        restore_best_weights = TRUE
      )
      # callback_learning_rate_scheduler(lr_schedule)
    ),
    # validation_split = 0.2
    validation_data = list(x = x_valid_img, y = y_valid),
    verbose = 1,
    shuffle = T,
    view_metrics = F
  )
