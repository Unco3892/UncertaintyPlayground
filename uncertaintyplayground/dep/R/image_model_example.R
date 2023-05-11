tensorflow::set_random_seed(1)
modL_img_v1 <- function(input_shape, hidden_img, act_output = "linear") {
  model_p_img_in <- create_cnn_prop(img_input_dim, regress = F)
  model_p_img_out <- model_p_img_in$output %>%
    # we can even add more layers here
    layer_dense(
      units = hidden_img,
      activation = act_output,
      name = "imag_z_t"
      # kernel_initializer = initializer_glorot_normal() #otherwise we have exploding gradients
      # kernel_initializer = initializer_random_normal(mean = 0, stddev = 1)
    )
  modL_p_img <- keras_model(inputs = model_p_img_in$input, outputs = model_p_img_out)
  modL_p_img %>%
    keras::compile(
      loss = "mse",
      weighted_metrics = list("mae", tfaddons::metric_rsquare())
      # optimizer_sgd(0.00001)
      # optimizer = optimizer_adamax(learning_rate =1.5e-3) #lr_schedule_img
      # optimizer = optimizer_sgd(learning_rate = lr_schedule_tab)
    )
}

# meta_model <- create_mlp(in_dim_tab, regress = F)
image_model <- modL_img_v1(in_dim_img, hidden_img)

image_model %>%
  fit(
    x = x_comp_img,
    y = mod_1[, (hidden_tab + 1):(hidden_tab + hidden_img)],
    epochs = 200,
    batch_size = 64,
    callbacks = callback_early_stopping(
      patience = 10,
      restore_best_weights = TRUE
    ),
    validation_split = 0.2,
    # validation_data = list(x = x_valid_img, y = y_valid),
    shuffle = T,
    verbose = 1,
    view_metrics = F,
    sample_weight = w_comp
  )

evaluate(image_model, x_comp_img, mod_1[, (hidden_tab + 1):(hidden_tab + hidden_img)])
