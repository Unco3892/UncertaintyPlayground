# model inspired from
# https://www.keep-current.dev/convolution-networks-on-tabular-data/

create_tabular_cnn <- function(num_columns = in_dim, n_outputs = out_dim, act_fun = "swish", dropout_rates = 0.1) {
  inp <- layer_input(shape = c(num_columns))
  x <- inp %>%
    layer_batch_normalization() %>%
    # layer_dropout(dropout_rates) %>%
    layer_dense(4096, activation = act_fun) %>%
    layer_reshape(c(256, 16)) %>%
    layer_batch_normalization() %>%
    layer_dropout(0.2) %>%
    layer_conv_1d(filters = 16, kernel_size = 5, activation = act_fun, use_bias = FALSE, padding = "SAME") %>%
    layer_average_pooling_1d(pool_size = 2)
  xs <- x
  x <- x %>%
    layer_batch_normalization() %>%
    layer_dropout(0.1) %>%
    layer_conv_1d(filters = 16, kernel_size = 3, activation = act_fun, use_bias = TRUE, padding = "SAME") %>%
    layer_batch_normalization() %>%
    layer_dropout(0.1) %>%
    layer_conv_1d(filters = 16, kernel_size = 3, activation = act_fun, use_bias = TRUE, padding = "SAME") %>%
    layer_multiply(xs) %>% # changed list(x, xs) to xs
    layer_max_pooling_1d(pool_size = 4, strides = 2) %>%
    layer_flatten() %>%
    layer_batch_normalization() %>%
    layer_activation(act_fun)
  out <- x %>% layer_dense(n_outputs, "linear")
  model <- keras_model(inputs = inp, outputs = out)
  return(model) # changed x to model
}

# a_model %>% compile(
#   # optimizer = optimizer_sgd(learning_rate = lr_schedule),
#   # optimizer = optimizer_adamax(0.07), #this goes well with relu
#   optimizer = optimizer_adam(0.006), # best with swish
#   # optimizer = "sgd",
#   loss = "mse",
#   # weighted_metrics = list(),
#   weighted_metrics = list(metric_root_mean_squared_error(name = "rmse"), tfaddons::metric_rsquare())
# )
#
# a_model %>% fit(
#   x_train,
#   y_train,
#   validation_split = 0.2,
#   callbacks = list(
#     callback_early_stopping(
#       monitor = "val_rmse",
#       patience = 50,
#       restore_best_weights = TRUE
#     )
#   ),
#   epochs = 1000,
#   verbose = 1
# )

# especially neural network with cnn activation functions
# postResample(pred = predict(a_model, x_train, verbose = 0), obs = y_train)
# postResample(pred = predict(a_model, x_test, verbose = 0), obs = y_test)

# a_model %>% compile(
#   # optimizer = optimizer_sgd(learning_rate = lr_schedule),
#   # optimizer = optimizer_adamax(0.07), #this goes well with relu
#   optimizer = optimizer_adam(0.006), # best with swish
#   # optimizer = "sgd",
#   loss = "mse",
#   # weighted_metrics = list(),
#   weighted_metrics = list(metric_root_mean_squared_error(name = "rmse"), tfaddons::metric_rsquare())
# )

#-------------------------------------------------------------------------------
# # try with swish and relu
# tensorflow::set_random_seed(1)
# # a_model <- create_tabular_cnn(num_columns = ncol(x_train), n_outputs = ncol(y_train))
# a_model <- create_tabular_cnn()
#
# a_model %>% compile(
#   # optimizer = optimizer_sgd(learning_rate = lr_schedule),
#   optimizer = optimizer_adam(0.0003), # best with swish
#   loss = "mse",
#   # weighted_metrics = list(),
#   weighted_metrics = list(metric_root_mean_squared_error(name = "rmse"), tfaddons::metric_rsquare())
# )
#
# a_model %>% fit(
#   x_comp,
#   mod_1,
#   validation_split = 0.2,
#   callbacks = list(
#     callback_early_stopping(
#       monitor = "val_rmse",
#       patience = 20,
#       restore_best_weights = TRUE
#     )
#   ),
#   epochs = 1000,
#   # use_multiprocessing = T,
#   verbose = 1,
#   # verbose = 0,
#   sample_weight = w_comp
# )
#
# modL %>% evaluate(x_comp, mod_1)
# a_model %>% evaluate(x_comp, mod_1)

# library(mgcv)
# p <- data.frame(x_comp, mod_1[,1:4])
# fff <- gam(
#   list(
#     X1 ~ s(living_space) + s(rooms) + s(lat) + s(lon),
#     X2 ~ s(living_space) + s(rooms) + s(lat) + s(lon),
#     X3 ~ s(living_space) + s(rooms) + s(lat) + s(lon),
#     X4 ~ s(living_space) + s(rooms) + s(lat) + s(lon)
#     # X5 ~ living_space + rooms + lat + lon
#   ),
#   data = p,
#   family = mvn(d = 4)
# )
#
# postResample(pred = predict(fff, newx = x_comp), obs = mod_1[,1:4])


# variation of gam with three functions, images, structured and a combination of the two

#-------------------------------------------------------------------------------
# weighted linear model

# p <- data.frame(x_comp, mod_1)
#
# fff <- lm(cbind(X1,X2,X3,X4,X5) ~ living_space + rooms + lat + lon, data = p, weights = w_comp)
# summary(fff)
#
# fff <- lm(cbind(X1,X2,X3,X4,X5) ~ living_space + rooms + lat + lon, data = p, weights = w_comp)
# summary(fff)
#
# postResample(predict(fff, p), obs = mod_1)
# postResample(predict(modL, x_comp), obs = mod_1)
