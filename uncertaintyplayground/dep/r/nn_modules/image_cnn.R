#####################################
### script for the cnn image model###
#####################################

# defining the filters and the dimension
model_filters <- seq_along(1:3)
channel_dim <- -1L

# defining the main property function
create_cnn_prop <- function(image_input = img_input_dim,
                            model_filters = c(16, 32, 64),
                            regress = FALSE) {
  # input layer
  inputs <-
    layer_input(shape = img_input_dim, name = "cnn_prop_input_layer")

  for (i in seq_along(model_filters)) {
    # if this is the first CONV layer then set the input
    # appropriately
    if (i == 1) {
      x <- inputs
    }

    # CONV => RELU => BN => POOL
    x %<>%
      layer_conv_2d(model_filters[i],
        c(3, 3),
        padding = "same",
        activation = "relu"
      ) %>%
      layer_batch_normalization(axis = channel_dim) %>%
      layer_max_pooling_2d(pool_size = c(2, 2))
  }
  # flatten the volume, then FC => RELU => BN => DROPOUT
  x %<>%
    layer_flatten() %>%
    layer_dense(16,
      name = "cnn_first_prop_dense",
      activation = "relu"
    ) %>%
    layer_batch_normalization(axis = channel_dim) %>%
    layer_dropout(0.5, name = "cnn_prop_dropout") %>%
    layer_dense(4,
      name = "cnn_last_prop_dense",
      activation = "relu"
    )
  # apply another FC layer, this one to match the number of nodes
  # coming out of the MLP

  if (regress == T) {
    x %<>%
      layer_dense(1, activation = "linear")
  }
  model <- keras_model(inputs, x)
  return(model)
}


# img_input_dim <- c(224, 224, 3)
# tensorflow::set_random_seed(1)
# image_model <- create_cnn_prop(img_input_dim, regress = T)
# lr_schedule <- learning_rate_schedule_exponential_decay(
#   0.001,
#   decay_steps = 5000,
#   decay_rate = 0.98,
#   staircase = TRUE
# )
#
# image_model %>% compile(
#   loss = "mse",
#   metrics = list(
#     metric_root_mean_squared_error(name = "rmse"),
#     tfaddons::metric_rsquare()
#   ),
#   optimizer = optimizer_sgd(learning_rate= lr_schedule)
#   # optimizer = optimizer_adamax(0.002) #lr_schedule
# )
#
# history <-
#   image_model %>%
#   fit(
#     x = x_train_img,
#     y = y_train,
#     epochs = 300,
#     batch_size = 8,
#     callbacks = callback_early_stopping(patience = 15,
#                                         restore_best_weights = TRUE),
#     # validation_split = 0.2
#     validation_data = list(x = x_valid_img, y = y_valid),
#     shuffle = T,
#     view_metrics = F
#   )
