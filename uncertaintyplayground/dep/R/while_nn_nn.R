# two neural networks (left and right)

# importing the data and setup
source("scripts/r/utilities/setup.R")

# import the activation functions
source("scripts/r/utilities/activation_funs.R")

# importing the inference functions
source("scripts/r/utilities/inference_funs.R")

#-------------------------------------------------------------------------------
# while loop setup
i <- 0 # iteration counter
continue <- TRUE
stopping_metric <- "Rsquared" # RMSE, #MAE
num_steps_without_improvement <- 0 # initialize a counter for number of steps without improvement
stopping_patience <- 5 # number of steps for stopping if no improvement
max_it <- 100
hidden <- 5
R <- 5

#-------------------------------------------------------------------------------
# each time we should take a different sample of the data to try this on

tensorflow::set_random_seed(1)
start_time <- Sys.time()
while (continue) {
  # inside the while loop for the iteration
  i <- i + 1

  if (i == 1) {
    ## variable definitions
    ### z_R ## N x R
    ### w_R ## N x R
    ### y_R ## N x R
    ### modL_p ## left hand side model (p for prime)
    ### modR_p ## right hand side model (p for prime)

    # get the dimensions
    set.seed(10)
    d <- ncol(x_train) + 1
    hidden <- 4

    # define the `z_R` array
    z_R <- array(NA, c(train_smp_size, hidden, R))

    # define left model and generate the weights based on the number of nodes
    modL_p <- matrix(rnorm(d * hidden), d, hidden)
    # make prediction with the left model
    z_t <- cbind(1, x_train) %*% modL_p

    # assign the weights for the model later on
    x_comp <- NULL ## X replicated R times NR x d
    y_comp <- NULL ## y replicated R times NR x 1
    for (r in 1:R) {
      x_comp <- rbind(x_comp, x_train)
      y_comp <- rbind(y_comp, y_train)
    }

    # define the performance registration loops
    modelL_perf <- list()
    modelR_perf <- list()
    train_perf <- list()
    valid_perf <- list()

    # define the learning rate for the neural networks
    lr_schedule <- learning_rate_schedule_exponential_decay(
      0.01,
      decay_steps = 100000,
      decay_rate = 0.96,
      staircase = TRUE
    )
  } else {
    # make prediction with the left model
    z_t <-
      predict(modL_p,
        x_train,
        verbose = 0,
        use_multiprocessing = T
      )
  }

  ## simulate the z_tilde -> use modL_p
  for (l in 1:hidden) {
    for (r in 1:R) {
      z_R[, l, r] <-
        rnorm(
          n = train_smp_size,
          mean = z_t[, l],
          sd = 1
        ) ## z_R is N x L x R
    }
  }

  # we can apply an activation function to `z_R` now or before (to `z_t`)
  # e.g.
  # z_R <- relu(z_R)
  # z_R <- tanh(z_R)
  # z_R <- swish(z_R)

  # we move to the right model
  if (i == 1) {
    # define the right model
    modR_p <- as.matrix(rnorm(hidden + 1))
    ## Generate the y_tilde
    y_R <- apply(z_R, 3, function(p) {
      cbind(1, p) %*% modR_p
    }) # R is the 3rd dimension of `z_R`, ## y_R (N x R)
  } else {
    # the issue is the weight's parameter in `lm` if using `predict()` -> to be discussed with Marc-O
    y_R <- apply(z_R, 3, function(p) {
      predict(
        modR_p,
        p,
        verbose = 0,
        use_multiprocessing = T
      )
    }) # R is the 3rd dimension of `z_R`, ## y_R (N x R)
  }

  ## Compute the weights -> apply the density of modR_p
  w_R <- apply(y_R, 2, function(y_estimate) {
    dnorm(y_train, mean = y_estimate, sd = 1)
  }) # R is the 2nd dimension

  ### just not sure about my way of doing `dnrom` vs yours
  denominator <- apply(w_R, 1, sum)

  ### operation sweep does the same
  w_R <-
    w_R / denominator ## standardized w_R by their row-wise sums

  ## unfold w_R from (N x R) to (NR x 1) vector
  w_comp <- c(w_R)

  # define the neural network parameters

  ## optimize Q-function for the left model
  # use a multi-variate neural network instead
  mod_1 <- apply(z_R, 2, function(p) {
    cbind(p)
  })

  in_dim <- dim(x_comp)[2]
  out_dim <- dim(mod_1)[2]

  # modL <- keras_model_sequential() %>%
  #   layer_dense(units = 1,
  #               activation = "linear",
  #               input_shape = in_dim) %>%
  #   layer_dense(units = out_dim, activation = "linear")

  tensorflow::set_random_seed(1)
  modL <- keras_model_sequential() %>%
    layer_dense(
      units = 256,
      activation = "tanh",
      input_shape = in_dim
    ) %>%
    layer_dropout(0.1) %>%
    layer_dense(
      units = 128,
      activation = "tanh"
    ) %>%
    layer_dense(units = 128, activation = "tanh") %>%
    layer_dense(units = 64, activation = "tanh") %>%
    layer_dense(units = out_dim, activation = "linear")

  # for adam, you have to use a larger learning rate (e.g. 0.01)
  # for sgd, you have to use a small learning rate (e.g. 0.001)
  # modL <- create_tabular_cnn(num_columns = in_dim, n_outputs = out_dim)

  modL %>% compile(
    # optimizer = optimizer_sgd(learning_rate = lr_schedule),
    optimizer = optimizer_adamax(0.0005),
    # optimizer = "adam",
    loss = "mse",
    # weighted_metrics = list(),
    weighted_metrics = list(metric_root_mean_squared_error(name = "rmse"), tfaddons::metric_rsquare())
  )

  modL %>% fit(
    x_comp,
    mod_1,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(
        monitor = "val_rmse",
        patience = 15,
        restore_best_weights = TRUE
      )
    ),
    epochs = 1000,
    # use_multiprocessing = T,
    # verbose = 1,
    verbose = 1,
    sample_weight = w_comp
  )

  tf$keras$Model$reset_metrics(modL)

  # make the prediction

  # assign this to the current model
  modL_p <- modL

  modelL_perf[[i]] <- modL_p

  ## optimize Q-function for the right model
  hidden_node <- apply(z_R, 2, c)
  # modR <- lm(y_comp ~ hidden_node , weights = w_comp)
  # modR <- nnet(hidden_node, y_comp, size = 5, weights = w_comp, linout = T,maxit = 1000)

  in_dim <- dim(hidden_node)[2]
  out_dim <- dim(y_comp)[2]

  # modR <- keras_model_sequential() %>%
  #   layer_dense(units = 1,
  #               activation = "linear",
  #               input_shape = in_dim) %>%
  #   layer_dense(units = out_dim, activation = "linear")

  modR <- keras_model_sequential() %>%
    layer_dense(
      units = 256,
      activation = "tanh",
      input_shape = in_dim
    ) %>%
    layer_dropout(0.1) %>%
    layer_dense(
      units = 128,
      activation = "tanh"
    ) %>%
    layer_dense(units = 128, activation = "tanh") %>%
    layer_dense(units = 64, activation = "tanh") %>%
    layer_dense(units = out_dim, activation = "linear")

  # modR <- create_tabular_cnn(num_columns = in_dim, n_outputs = out_dim)

  modR %>% compile(
    # optimizer = optimizer_sgd(learning_rate = lr_schedule),
    optimizer = optimizer_adamax(0.001),
    # optimizer = "adam",
    loss = "mse",
    # weighted_metrics = list(),
    weighted_metrics = list(metric_root_mean_squared_error(name = "rmse"), tfaddons::metric_rsquare())
  )

  modR %>% fit(
    hidden_node,
    y_comp,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(
        monitor = "val_rmse",
        patience = 15,
        restore_best_weights = TRUE
      )
    ),
    epochs = 1000,
    verbose = 0,
    # verbose = 1,
    # use_multiprocessing = T,
    sample_weight = w_comp
  )

  modR_p <- modR

  tf$keras$Model$reset_metrics(modR)

  # log the performance metric of the right side
  modelR_perf[[i]] <- modR_p

  # log the overall perpformance metrics (training and validation)
  train_perf[[i]] <-
    postResample(pred = infer_nn_nn(df_train), obs = y_train)
  valid_perf[[i]] <-
    postResample(
      pred = infer_nn_nn(df_valid, size = valid_smp_size),
      obs = y_valid
    )

  # printing diagnostics
  message(
    "Iteration: ",
    i,
    ", Train R2 & RMSE: ",
    round(train_perf[[i]]["Rsquared"], 3),
    " & ",
    round(train_perf[[i]]["RMSE"], 3),
    ", Validation R2 & RMSE: ",
    round(valid_perf[[i]]["Rsquared"], 3),
    " & ",
    round(valid_perf[[i]]["RMSE"], 3)
  )

  # early stopping defined
  if ((i > 2)) {
    if ((valid_perf[[i]][stopping_metric] > max(sapply(valid_perf[1:length(valid_perf) -
      1], "[[", stopping_metric)))) {
      num_steps_without_improvement <- 0 # reset the counter
    } else {
      num_steps_without_improvement <- num_steps_without_improvement + 1
    }
  }
  # we check if we have improved or not
  if (num_steps_without_improvement >= stopping_patience) {
    optimal_i <-
      which.max(sapply(valid_perf[1:length(valid_perf)], "[[", stopping_metric))
    modL_p <- modelL_perf[[optimal_i]]
    modR_p <- modelR_perf[[optimal_i]]
    # check if the counter has reached the threshold
    continue <- FALSE
  }

  # if we don't coverge, we can also break the loop
  if (i == max_it) {
    continue <- FALSE
  }
}

end_time <- Sys.time()
end_time - start_time


# ------------------------------------------------------------------------------
# measure this performance on both the training and test sets
# training set
lm_mod_train <- lm(Y ~ ., data = df_train)
postResample(pred = predict(lm_mod_train), obs = y_train)
postResample(pred = infer_nn_nn(df_train), obs = y_train)

# test set
postResample(
  pred = predict(lm_mod_train, newdata = df_test),
  obs = y_test
)
postResample(pred = infer_nn_nn(df_test, size = nrow(df_test)), obs = y_test)

#-------------------------------------------------------------------------------
# neural network and random forest
# neural network
in_dim <- dim(x_train)[2]
out_dim <- dim(y_train)[2]

tensorflow::set_random_seed(1)
modeLNN <- keras_model_sequential() %>%
  layer_dense(
    units = 256,
    activation = "tanh",
    input_shape = in_dim
  ) %>%
  layer_dense(
    units = 128,
    activation = "tanh"
  ) %>%
  layer_dense(units = 128, activation = "tanh") %>%
  layer_dense(units = 64, activation = "tanh") %>%
  layer_dense(units = out_dim, activation = "linear")

modeLNN %>% compile(
  # optimizer = optimizer_sgd(learning_rate = lr_schedule),
  optimizer = optimizer_adam(0.005),
  # optimizer = "sgd",
  loss = "mse",
  # weighted_metrics = list(),
  metrics = list(metric_root_mean_squared_error(name = "rmse"))
)

modeLNN %>% fit(
  x_train,
  y_train,
  validation_split = 0.2,
  callbacks = list(
    callback_early_stopping(
      monitor = "val_rmse",
      patience = 20,
      restore_best_weights = TRUE
    )
  ),
  epochs = 1000,
  verbose = 0
)

## check the performance on the training and test sets
postResample(pred = predict(modeLNN, x_train, verbose = 0), obs = y_train)
postResample(pred = predict(modeLNN, x_test, verbose = 0), obs = y_test)

# random forest
randomForest <-
  caret::train(Y ~ ., df_train, method = "rf")
randomForest <-
  caret::train(Y ~ .,
    df_train,
    method = "rf",
    trControl = trainControl(method = "none")
  )

## check the performance on the training and test sets
postResample(pred = predict(randomForest, x_train), obs = y_train)
postResample(pred = predict(randomForest, x_test), obs = y_test)
