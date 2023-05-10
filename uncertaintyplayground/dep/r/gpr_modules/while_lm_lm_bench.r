# Title: Multi-variate linear model (left) and random forest (right)
# ------------------------------------------------------------------------------

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
stopping_metric <- "Rsquared" # other metrics can be `RMSE` and `MAE`
stopping_patience <- 15 # number of steps for stopping if no improvement
max_it <- 2
hidden <- 50 # even better results with 19 hiddens (in addition to R=9)
R <- 100 # 9 with `rf` worked super well and even bit the linear model

# new parameter added for the standard deviation
custom_sd <- 0.5

#-------------------------------------------------------------------------------

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

    # get the dimensions
    set.seed(10)
    d <- ncol(x_train) + 1

    # define the `z_R` array
    z_R <- array(NA, c(train_smp_size, hidden, R))

    # define left model and generate the weights based on the number of nodes
    modL_p <- matrix(rnorm(d * hidden), d, hidden)
    # make prediction with the left model
    z_t <- cbind(1, x_train) %*% modL_p
    # z_t <- tanh(z_t)

    # z_t <- scale(z_t)
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

    # start the step counter
    num_steps_without_improvement <- 0 # initialize a counter for number of steps without improvement
  } else {
    # make prediction with the left model
    z_t <-
      predict(
        modL_p,
        data.frame(x_train)
      )
    # z_t <- scale(z_t)
  }

  ## simulate the z_tilde -> use modL_p
  for (l in 1:hidden) {
    for (r in 1:R) {
      z_R[, l, r] <-
        rnorm(
          n = train_smp_size,
          mean = z_t[, l],
          sd = custom_sd
        ) ## z_R is N x L x R
    }
  }

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
        data.frame(p)
      )
    }) # R is the 3rd dimension of `z_R`, ## y_R (N x R)
  }

  ## Compute the weights -> apply the density of modR_p
  w_R <- apply(y_R, 2, function(y_estimate) {
    dnorm(y_train, mean = y_estimate, sd = 1)
  }) # R is the 2nd dimension

  ### just not sure about my way of doing `dnrom` vs yours
  denominator <- apply(w_R, 1, sum)

  # we check that there is 0/0 division in the first iteration
  if (i == 1) {
    # a very small smoothing added
    epsilon <- 1e-12
    if (any(denominator < epsilon)) {
      smoothed_denominator <- ifelse(denominator < epsilon, denominator + epsilon, denominator)
      w_R <- w_R / smoothed_denominator
    }
  } else {
    ### operation sweep does the same
    w_R <- w_R / denominator ## standardized w_R by their row-wise sums
  }

  ### operation sweep does the same
  # w_R <- w_R / denominator ## standardized w_R by their row-wise sums

  ## unfold w_R from (N x R) to (NR x 1) vector
  w_comp <- c(w_R)

  # define the neural network parameters

  ## optimize Q-function for the left model
  # use a multi-variate neural network instead
  mod_1 <- apply(z_R, 2, function(p) {
    cbind(p)
  })

  # create the dataframe to be used by the model
  data_left_model <- data.frame(x_comp, mod_1)

  modL <-
    # we use this technique to make sure that we approach the multi-variate
    # situation in a dynamic way with all the inputs automatically chosen
    lm(
      as.formula(str_c(
        "cbind(",
        paste0("X", 1:hidden, collapse = ","),
        ")~",
        paste0(names(data_left_model)[!(names(data_left_model) %in% str_c("X", 1:hidden))], collapse = "+")
      )),
      data = data_left_model,
      weights = w_comp
    )

  # print(evaluate(modL_p, x = x_comp, y = mod_1, sample_weight = matrix(w_comp)))
  # print(evaluate(modL_p, x = x_comp, y = mod_1, sample_weight = w_comp))
  # just print it the first time as afterwards it can become disturbing
  if (i == 1) {
    cat("Left model (lm):\n")
  }

  print(postResample(predict(modL, data_left_model), obs = mod_1))

  # assign this to the current model
  modL_p <- modL
  modelL_perf[[i]] <- modL_p

  ## optimize Q-function for the right model
  hidden_node <- apply(z_R, 2, c)

  ## random forest for the left model
  set.seed(1)
  modR <- caret::train(
    y_comp ~ .,
    data.frame(hidden_node, y_comp),
    # method = "Rborist", #rf
    method = "lm",
    # Rborist
    trControl = trainControl(method = "none"),
    weights = w_comp
  )

  # just print it the first time as afterwards it can become disturbing
  if (i == 1) {
    cat("Right model (rf):\n")
  }
  print(postResample(predict(modR), y_comp))

  # assign this to the right model
  modR_p <- modR
  modelR_perf[[i]] <- modR_p

  # log the overall perpformance metrics (training and validation)
  train_perf[[i]] <-
    postResample(pred = infer_multivlm_caret(df_train), obs = y_train)
  valid_perf[[i]] <-
    postResample(
      pred = infer_multivlm_caret(df_valid, size = valid_smp_size),
      obs = y_valid
    )

  # printing diagnostics
  message(
    "Iteration: ",
    i,
    ", Overall Train R2 & RMSE: ",
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
