#######################################
### script for the mlp tabular model###
#######################################
library(keras)
library(tfaddons)

# defining the meta model (MLP)
create_mlp <- function(dim, regress = FALSE) {
  # input layer
  input_tensor <- layer_input(shape = dim, name = "mlp_meta_input_layer")
  output_tensor <-
    input_tensor %>%
    layer_dense(
      units = 256,
      activation = "tanh", # relu
      input_shape = dim
    ) %>%
    layer_dropout(0.1) %>% # adding this dropout improved the performance drastically
    layer_dense(
      units = 128,
      activation = "tanh" # relu
    ) %>%
    # layer_dropout(0.1) %>%
    layer_dense(
      units = 128,
      activation = "tanh" # relu
    ) %>%
    layer_dense(
      units = 64,
      activation = "tanh" # relu
    )
  if (regress == T) {
    output_tensor %<>%
      layer_dense(1, activation = "linear", name = "mlp_meta_regression")
  }
  model <- keras_model(input_tensor, output_tensor)
  return(model)
}

# # creating the model
meta_model <- create_mlp(c(4), T)
a_seed <- 2021
opt <- optimizer_adamax(lr = 1.5e-3)

meta_model %>% compile(
  loss = "mse", # "mae" is the second best with adamax optimizer
  # best result comes from log_cosh with optimizer_adamax(lr=1.5e-3)
  metrics = list(
    "mae",
    tf$keras$metrics$RootMeanSquaredError(name = "rmse"),
    tfaddons::metric_rsquare()
  ),
  optimizer = opt
)
#
history <- meta_model %>%
  fit(
    x_train,
    y_train,
    epochs = 100,
    batch_size = 64,
    callbacks = callback_early_stopping(
      patience = 10,
      restore_best_weights = TRUE
    ),
    validation_split = 0.1,
    view_metrics = FALSE
  )
#
# meta_model %>%
#   evaluate(x = list(test_x),
#            y = test_y)
