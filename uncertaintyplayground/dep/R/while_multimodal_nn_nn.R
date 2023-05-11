# ------------------------------------------------------------------------------
# Title: Multi-modal neural network (left) and random forest (right)
# ------------------------------------------------------------------------------

# turn off the warnings
options(warnings = -1)

# general setting for the data
set.seed(1)
N <- 1000
img_h_w <- c(61, 61)
rgb <- T
img_input_dim <- c(img_h_w, ifelse(rgb, 3, 1))
train_size <- 0.5
valid_size <- 0.25
pred_modality_togthr <- T # if the intention is to predict the two seperately

# while loop setup
i <- 0 # iteration counter
continue <- TRUE
stopping_metric <- "Rsquared" # other metrics can be `RMSE` and `MAE`
stopping_patience <- 5 # number of steps for stopping if no improvement
max_it <- 200
hidden_tab <- 5 # even better results with 19 hidden_tabs (in addition to R=9)
hidden_img <- 10
R <- 5
# R <- 2 #9 with `rf` worked super well and even bit the linear model

# new parameter added for the standard deviation
custom_sd <- 1

#-------------------------------------------------------------------------------

# importing the data (lazyload) and multi-setup
source("scripts/r/utilities/setup_multimodal.R")
# import the activation functions
source("scripts/r/utilities/activation_funs.R")
# importing the inference functions
source("scripts/r/utilities/inference_funs.R")
# load the model definitions
source("scripts/r/models/image_cnn.R")
source("scripts/r/models/tab_mlp.R")

#-------------------------------------------------------------------------------

x_comp_tab <- NULL ## X replicated R times NR x d
x_comp_img <- NULL ## X replicated R times NR x d
y_comp <- NULL ## y replicated R times NR x 1

for (r in 1:R) {
  x_comp_img <- abind(x_comp_img, x_train_img, along = 1) # this is similar to rbind in the context of array
  x_comp_tab <- rbind(x_comp_tab, x_train_tab)
  y_comp <- rbind(y_comp, y_train)
}

#-------------------------------------------------------------------------------

tensorflow::set_random_seed(1)
# applying the while loop to each step
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

    set.seed(10)
    d_tab <- ncol(x_train_tab) + 1
    d_img <- length(x_train_img) + 1 # N * H * W * RGB + 1

    # define the `z_R` arrays (N x HIDDEN NODES FOR THAT MODALITY)
    z_R_tab <- array(NA, c(train_smp_size, hidden_tab, R))
    z_R_img <- array(NA, c(train_smp_size, hidden_img, R))

    # define the tabular and image random weights (between -3 and 3)
    modL_p_tab <- matrix(rnorm(d_tab * hidden_tab), d_tab, hidden_tab)
    # create the image model from `helpers.R` (between -1 and 1)
    modL_p_img <- create_random_nn_pred(img_input_dim, hidden_img = hidden_img, act_output = "tanh")

    # if you want to also place the output of the neural network between -3 and 3,
    # we can multiply the output by 3

    # HOW SHOULD
    # make predictions for the tabular side
    z_t_tab <- cbind(1, x_train_tab) %*% modL_p_tab
    # make predictions with the image model
    z_t_img <- predict(modL_p_img, x_train_img)

    # later bring the simulation here

    # define the performance registration loops
    modelL_perf <- list()
    modelR_perf <- list()
    train_perf <- list()
    valid_perf <- list()

    # start the step counter
    num_steps_without_improvement <- 0 # initialize a counter for number of steps without improvement

    # define the learning rate scheduler for the neural networks
    lr_schedule_tab <- learning_rate_schedule_exponential_decay(
      0.01,
      decay_steps = 100000,
      decay_rate = 0.96,
      staircase = TRUE
    )

    lr_schedule_img <- learning_rate_schedule_exponential_decay(
      0.001,
      decay_steps = 5000,
      decay_rate = 0.98,
      staircase = TRUE
    )
  } else {
    # make prediction with the left model
    z_t <-
      predict(modL_p,
        list(x_train_tab, x_train_img),
        verbose = 0
      )
  }

  # we can apply an activation function to `z_t` now or after
  # e.g. z_t <- relu(z_t)

  ## simulate the z_tilde for tabular data -> use modL_p_tab
  for (l in 1:hidden_tab) {
    for (r in 1:R) {
      z_R_tab[, l, r] <- rnorm(
        n = train_smp_size,
        mean = `if`((i > 1 && # assuming z_t_tab is on the left of z_t
          pred_modality_togthr), z_t[, l], z_t_tab[, l]),
        sd = 1
      ) ## z_R_tab is N x L x R
    }
  }

  ## simulate the z_tilde for image data -> use modL_p_img
  for (l in 1:hidden_img) {
    for (r in 1:R) {
      z_R_img[, l, r] <- rnorm(
        n = train_smp_size,
        mean = `if`((i > 1 && # assuming z_t_img is on the right of z_t
          pred_modality_togthr), z_t[, l + hidden_tab], z_t_img[, l]),
        sd = 1
      ) ## z_R_tab is N x L x R
    }
  }

  # we move to the right model
  if (i == 1) {
    # define the right model
    modR_p <- as.matrix(rnorm(hidden_tab + hidden_img + 1))

    # join all the hidden nodes together
    z_R <- abind(list(z_R_tab, z_R_img), along = 2)

    ## Generate the y_tilde
    y_R <- apply(z_R, 3, function(p) {
      cbind(1, p) %*% modR_p
    }) # R is the 3rd dimension of `z_R`, ## y_R (N x R)
  } else {
    # join all the hidden nodes together
    z_R <- abind(list(z_R_tab, z_R_img), along = 2)

    # the issue is the weight's parameter in `lm` if using `predict()` -> to be discussed with Marc-O
    y_R <- apply(z_R, 3, function(p) {
      predict(
        modR_p, p
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

  ## optimize Q-function for the left model
  # use a multi-variate neural network instead
  mod_1 <- apply(z_R, 2, function(p) cbind(p))

  in_dim_tab <- dim(x_comp_tab)[2]
  in_dim_img <- dim(x_comp_img)[-1]
  out_dim <- dim(mod_1)[2]

  # tensorflow::set_random_seed(1)

  # define the meta and image models
  meta_model <- create_mlp(in_dim_tab, regress = F)
  image_model <- create_cnn_prop(in_dim_img, regress = F)

  # combine the layers
  combined_modalities <- layer_concatenate(list(meta_model$output, image_model$output)) %>%
    # we can even add more layers here
    layer_dense(
      units = out_dim,
      activation = "linear",
      name = "intermediate_pred"
    )

  # define the final model
  modL <- keras_model(
    inputs = list(meta_model$input, image_model$input),
    outputs = combined_modalities
  )

  # compile the model
  modL %>%
    keras::compile(
      loss = "mse",
      weighted_metrics = list("mae", tfaddons::metric_rsquare()),
      # optimizer = optimizer_adamax(learning_rate =1.5e-3) #lr_schedule_img
      optimizer = optimizer_sgd(learning_rate = lr_schedule_tab)
    )

  modL %>%
    fit(
      x = list(mlp_meta_input_layer = x_comp_tab, cnn_prop_input_layer = x_comp_img),
      y = mod_1,
      epochs = 200,
      batch_size = 64,
      callbacks = callback_early_stopping(
        patience = 10,
        restore_best_weights = TRUE
      ),
      validation_split = 0.2,
      # validation_data = list(x = x_valid_img, y = y_valid),
      shuffle = T,
      view_metrics = F,
      sample_weight = w_comp
    )


  tf$keras$Model$reset_metrics(modL)

  if (i == 1) {
    cat("Left model (Multi-nn):\n")
  }

  print(postResample(predict(modL, list(mlp_meta_input_layer = x_comp_tab, cnn_prop_input_layer = x_comp_img)), obs = mod_1))

  # assign this to the current model
  modL_p <- modL
  modelL_perf[[i]] <- modL_p

  ## optimize Q-function for the right model
  hidden_node <- apply(z_R, 2, c)

  in_dim <- dim(hidden_node)[2]
  out_dim <- dim(y_comp)[2]

  tensorflow::set_random_seed(1)
  modR <- keras_model_sequential() %>%
    layer_dense(
      units = 1,
      activation = "linear",
      input_shape = in_dim
    ) %>%
    layer_dense(units = out_dim, activation = "linear")

  # modR <- keras_model_sequential() %>%
  #   layer_dense(units = 256,
  #               activation = "tanh",
  #               input_shape = in_dim) %>%
  #   layer_dropout(0.1) %>%
  #   layer_dense(units = 128,
  #               activation = "tanh") %>%
  #   layer_dense(units = 128, activation = "tanh") %>%
  #   layer_dense(units = 64, activation = "tanh") %>%
  #   layer_dense(units = out_dim, activation = "linear")

  # modR <- create_tabular_cnn(num_columns = in_dim, n_outputs = out_dim)

  modR %>% compile(
    # optimizer = optimizer_sgd(learning_rate = lr_schedule_tab),
    optimizer = optimizer_sgd(),
    # optimizer = optimizer_adamax(0.001),
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
    epochs = 500,
    # verbose = 0,
    verbose = 1,
    # use_multiprocessing = T,
    sample_weight = w_comp
  )

  # just print it the first time as afterwards it can become disturbing
  if (i == 1) {
    cat("Right model (nn):\n")
  }
  print(postResample(predict(modR, hidden_tab_node), y_comp))

  modR_p <- modR

  tf$keras$Model$reset_metrics(modR)

  # log the performance metric of the right side
  modelR_perf[[i]] <- modR_p

  # # log the overall perpformance metrics (training and validation)
  # train_perf[[i]] <-
  #   # have to adapt the inference functions so that it accepts images
  #   postResample(pred = infer_nn_nn(df_train), obs = y_train)
  # valid_perf[[i]] <-
  #   postResample(pred = infer_nn_nn(df_valid, size = valid_smp_size),
  #                obs = y_valid)

  # # printing diagnostics
  # message(
  #   "Iteration: ",
  #   i,
  #   ", Train R2 & RMSE: ",
  #   round(train_perf[[i]]['Rsquared'], 3),
  #   " & ",
  #   round(train_perf[[i]]['RMSE'], 3),
  #   ", Validation R2 & RMSE: ",
  #   round(valid_perf[[i]]['Rsquared'], 3),
  #   " & ",
  #   round(valid_perf[[i]]['RMSE'], 3)
  # )
  #
  # # early stopping defined
  # if ((i > 2)) {
  #   if ((valid_perf[[i]][stopping_metric] > max(sapply(valid_perf[1:length(valid_perf) -
  #                                                                 1], "[[", stopping_metric)))) {
  #     num_steps_without_improvement <- 0 # reset the counter
  #   } else{
  #     num_steps_without_improvement <- num_steps_without_improvement + 1
  #   }
  # }
  # # we check if we have improved or not
  # if (num_steps_without_improvement >= stopping_patience) {
  #   optimal_i <-
  #     which.max(sapply(valid_perf[1:length(valid_perf)], "[[", stopping_metric))
  #   modL_p <- modelL_perf[[optimal_i]]
  #   modR_p <- modelR_perf[[optimal_i]]
  #   # check if the counter has reached the threshold
  #   continue <- FALSE
  # }

  # if we don't coverge, we can also break the loop
  if (i == max_it) {
    continue <- FALSE
  }
}

end_time <- Sys.time()
end_time - start_time


# CHANGE THE BENCHMARKING TO MARK IMAGES VS NO IMAGES
# ------------------------------------------------------------------------------
# train a random forest and linear regression on the original data for comparison
randomForest <- caret::train(Y ~ ., df_train, method = "rf", trControl = trainControl(method = "none"))
lm_mod_train <- lm(Y ~ ., data = df_train)

#-------------------------------------------------------------------------------
# measuring the performance on the training and test sets
# create the variables needed for the model performances
model_list <- list(lm_mod_train, NA, randomForest) # we can replace our approach with an NA since it's a function
model_names <- c("simple_lm", paste0("em_lm_rf_", "h", hidden, "_r", R), "simple_rf")
names(model_list) <- model_names
subset_list <- list(df_train, df_valid, df_test)
subset_names <- c(paste0("train_", nrow(df_train), "n"), paste0("valid_", nrow(df_valid), "n"), paste0("test_", nrow(df_test), "n"))
names(subset_list) <- subset_names
compare_models <- as.list(rep(NA, length(model_list)))
names(compare_models) <- subset_names

# make a loop to check apply each model to eat subset and record the results
for (i in 1:length(subset_list)) {
  sub <- subset_list[[i]]
  subset_perf <- list()
  for (j in 1:length(model_list)) {
    mod <- model_list[[j]]
    if (grepl("em_lm_rf", model_names[j], fixed = TRUE)) {
      mod_pred <- infer_multilm_caret(sub, size = nrow(sub))
    } else {
      mod_pred <- predict(mod, newdata = sub)
    }
    pred_metric <- postResample(pred = mod_pred, obs = pull(sub, Y))
    subset_perf[[j]] <-
      tibble(
        R2 = pred_metric[2],
        RMSE = pred_metric[1],
        MAE = pred_metric[3],
        model = model_names[j]
      )
  }
  compare_models[[i]] <- bind_rows(subset_perf)
}
compare_models

# ------------------------------------------------------------------------------
# # additional code if you like to test each model individually
#
# # training set results
# postResample(pred = predict(lm_mod_train), obs = y_train)
# postResample(pred = infer_multilm_caret(df_train), obs = y_train)
# postResample(pred = predict(randomForest, x_train), obs = y_train)
#
# # validation set
# postResample(pred = predict(lm_mod_train, newdata = df_valid), obs = y_valid)
# postResample(pred = infer_multilm_caret(df_valid, size = nrow(df_valid)), obs = y_valid)
# postResample(pred = predict(randomForest, x_valid), obs = y_valid)
#
# # test set
# postResample(pred = predict(lm_mod_train, newdata = df_test),obs = y_test)
# postResample(pred = infer_multilm_caret(df_test, size = nrow(df_test)), obs = y_test)
# postResample(pred = predict(randomForest, x_test), obs = y_test)
