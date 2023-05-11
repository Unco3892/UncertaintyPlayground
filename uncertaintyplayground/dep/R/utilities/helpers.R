# helper functions
# ------------------------------------------------------------------------------
# keras functions
read_single_img <- function(image_path, target_size = c(224, 224)) {
  keras::image_load(path = image_path, target_size = target_size) %>%
    image_to_array() %>%
    `/`(255)
  # %>% array_reshape(c(1, dim(.)))
}

load_imgs_from_dir <- function(paths,
                               target_size = c(224, 224)) {
  if (!is.list(paths)) {
    warning("paths was not a list, converted to a list")
    paths <- as.list(paths)
  }
  if (length(paths) == 1) {
    stop("Length of paths is 1")
  }
  array_lists <-
    lapply(paths, read_single_img, target_size = target_size)
  return(abind(array_lists, along = 0))
}

# ------------------------------------------------------------------------------
# making a random prediction with a neural network
# using our own initial model for initialization
create_random_nn_pred_v3 <- function(input_shape, hidden_img, act_output = "tanh") {
  model_p_img_in <- create_cnn_prop(input_shape, regress = F)
  model_p_img_out <- model_p_img_in$output %>%
    # we can even add more layers here
    layer_dense(
      units = hidden_img,
      activation = act_output,
      name = "first_pred",
      # kernel_initializer = initializer_glorot_normal() #otherwise we have exploding gradients
      kernel_initializer = initializer_random_normal(mean = 0, stddev = 1)
    )
  modL_p_img <- keras_model(inputs = model_p_img_in$input, outputs = model_p_img_out)
  modL_p_img
}

create_random_nn_pred_v1 <- function(input_shape, hidden_img, act_output = "linear") {
  model <- keras_model_sequential() %>%
    layer_flatten(input_shape = input_shape) %>%
    layer_dense(
      units = hidden_img,
      activation = act_output,
      kernel_initializer = initializer_random_normal(mean = 0, stddev = 1)
      # kernel_initializer = initializer_glorot_normal() #otherwise we have exploding gradients
    )
  model
}

#  single pass of multi-modal neural network
# create_random_nn_pred_v2 <- function(input_shape, hidden_img, act_output = "linear") {
#   model <- keras_model_sequential() %>%
#     layer_flatten(input_shape = input_shape) %>%
#     layer_dense(
#       units = hidden_img,
#       activation = act_output,
#       # kernel_initializer = initializer_random_normal(mean = 0, stddev = 1)
#       kernel_initializer = initializer_glorot_normal() #otherwise we have exploding gradients
#     )
#   model
# }


# create_random_cnn_pred <- function(input_shape, hidden_img, act_output = "linear") {
#   model <- keras_model_sequential() %>%
#     layer_conv_2d(
#       filters = 32,
#       kernel_size = c(3,3),
#       input_shape = input_shape,
#       activation = "relu",
#       kernel_initializer = initializer_random_normal(mean = 0, stddev = 1)
#     ) %>%
#     layer_max_pooling_2d(pool_size = c(2, 2)) %>%
#     layer_flatten() %>%
#     layer_dense(
#       units = hidden_img,
#       activation = act_output,
#       kernel_initializer = initializer_random_normal(mean = 0, stddev = 1)
#     )
#   model
# }


# ------------------------------------------------------------------------------
# other functions that may be necessary later
# alternatives
# generator <-
#   image_data_generator(rescale = 1 / 255)
#
# a <- flow_images_from_directory(
#   file.path(here::here(), "data/sample/train"),
#   batch_size= 32,
#   generator = generator,
#   target_size = c(224, 224),
# )
#
# train <- flow_images_from_dataframe(
#   sample_tibble,
#   x_col = "image",
#   y_col = "label",
#   batch_size= 32L,
#   generator = generator,
#   target_size = c(224, 224),
#   class_mode = "raw",
#   drop_duplicates = FALSE)
