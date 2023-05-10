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
stopping_patience <- 20 # number of steps for stopping if no improvement
max_it <- 400
R <- 100 # 9 with `rf` worked super well and even bit the linear model

# new parameter added for the standard deviation
custom_sd <- 1

# we compute the number of rows of the data
n_features <- ncol(x_train)
n_z_outcomes <- c(5, 20, 2, 2)
# n_z_outcomes <- rep(1,n_features)
hidden <- sum(n_z_outcomes)

#-------------------------------------------------------------------------------

## simulate the z_tilde -> use modL_p
sample_z_r <- function(n_hiddens, z_means, a_size = train_smp_size, sampling_R = R, desired_sd = custom_sd) {
  # define the `z_R` array
  desired_z_R <- array(NA, c(a_size, n_hiddens, sampling_R))
  for (l in 1:n_hiddens) {
    for (r in 1:sampling_R) {
      desired_z_R[, l, r] <-
        rnorm(
          n = a_size,
          mean = z_means[, l],
          sd = desired_sd
        ) ## desired_z_R is N x L x sampling_R
    }
  }
  return(desired_z_R)
}

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
    
    # define left model and generate the weights based on the number of nodes
    modL_p <- matrix(rnorm(d * hidden), d, hidden)
    # make prediction with the left model
    z_t <- cbind(1, x_train) %*% modL_p
    
    # only during the first iteration this is needed because in other iterations, it is alreay
    # the case with the predictions
    start_index <- c(1, cumsum(n_z_outcomes[-length(n_z_outcomes)]) + 1)
    end_index <- cumsum(n_z_outcomes)
    # z_t <- mapply(function(start, end) z_t[, start:end, drop = F], start_index, end_index)
    
    z_t <- lapply(1:length(n_z_outcomes), function(i) z_t[, start_index[i]:end_index[i], drop = FALSE])
    
    
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
    x_train_new <- as.data.frame(x_train)
    z_t <- lapply(1:n_features, function(m) {
      predict(modL_p[[m]], newdata = x_train_new) # here we assume one input node, one/multiple output nodes
    }) # I could have done also x_train_new[,m, drop =F]
    
    # z_t <- scale(z_t)
  }
  
  z_R_sep <- list()
  for (p in 1:length(n_z_outcomes)) {
    if (!is.matrix(z_t[[p]])) {
      z_t[[p]] <- as.matrix(z_t[[p]])
    }
    z_R_sep[[p]] <- sample_z_r(n_z_outcomes[p], z_means = z_t[[p]])
  }
  
  # then we concatenate all the outcomes
  z_R <- abind::abind(z_R_sep, along = 2)
  
  # we move to the right model
  if (i == 1) {
    # define the right model
    modR_p <- as.matrix(rnorm(sum(n_z_outcomes) + 1))
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
  
  # calculate the denominator
  denominator <- apply(w_R, 1, sum)
  
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
  
  ## unfold w_R from (N x R) to (NR x 1) vector
  w_comp <- c(w_R)
  
  # define the neural network parameters
  
  # define multiple linear regressions (some univariate and some multi-variate)
  
  ## optimize Q-function for the left model
  # use a multi-variate neural network instead
  # mod_1 <- apply(z_R, 2, function(p) {
  #   cbind(p)
  # })
  # create the dataframe to be used by the model
  
  modL <- list()
  obs_list <- list()
  for (v in 1:n_features) {
    inputs_i <- x_comp[, v, drop = FALSE]
    outputs_i <- z_R_sep[[v]]
    mod_1 <- apply(outputs_i, 2, function(p) {
      cbind(p)
    })
    obs_list[[v]] <- mod_1
    data_left_model <- data.frame(inputs_i, mod_1)
    if (dim(outputs_i)[2] > 1) {
      formula <- str_c(
        "cbind(",
        paste0("X", 1:n_z_outcomes[v], collapse = ","),
        ")"
      )
      formula <-
        as.formula(str_c(formula, "~", paste0(names(data_left_model)[!(names(data_left_model) %in% str_c("X", 1:n_z_outcomes[v]))],
                                              collapse = "+"
        )))
    } else {
      formula <- as.formula(str_c(paste0(names(data_left_model)[!(names(data_left_model) %in% colnames(inputs_i))]),"~", colnames(inputs_i)))
    }
    
    # run the linear regression
    modL[[v]] <- lm(formula, data = data_left_model, weights = w_comp)
  }
  
  # just print it the first time as afterwards it can become disturbing
  if (i == 1) {
    cat("Left models (multiple lms):\n")
  }
  
  # we make a prediction but simply adding all the columns together
  pred_list <- do.call(cbind,lapply(modL, function(x) predict(x, newdata = as.data.frame(x_comp))))
  
  # maybe in my predictions I have to include sample weights?
  print(postResample(pred_list, obs = do.call(cbind, obs_list)))
  
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
    postResample(pred = infer_mlinear_lm(df_train), obs = y_train)
  valid_perf[[i]] <-
    postResample(
      pred = infer_mlinear_lm(df_valid),
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

# the performance is really awful, find out why


# ------------------------------------------------------------------------------
# plot performances
# train_perf_plt <- bind_rows(train_perf) %>%
#   mutate(subset = "train", iteration = row_number()) %>%
#   gather(key = "metric", value = "value", -c(subset, iteration))
# 
# valid_perf_plt <- bind_rows(valid_perf) %>%
#   mutate(subset = "valid", iteration = row_number()) %>%
#   gather(key = "metric", value = "value", -c(subset, iteration))
# 
# perf_plot <- bind_rows(
#   train_perf_plt,
#   valid_perf_plt
# ) %>%
#   ggplot(aes(
#     x = iteration,
#     y = value,
#     color = subset,
#     group = subset
#   )) +
#   geom_point() +
#   geom_line() +
#   theme_light() +
#   facet_wrap(~metric,
#              nrow = 3,
#              ncol = 1,
#              scales = "free_y"
#   ) +
#   scale_x_continuous(breaks = seq(
#     from = 1,
#     to = length(train_perf),
#     by = 3
#   )) +
#   ggtitle(paste0("Model multi-variate lm + rf with ", "h=", hidden, " R=", R)) +
#   geom_vline(aes(xintercept = optimal_i, color = "Optimal_i"),
#              linetype = "dashed"
#   ) +
#   theme(
#     legend.position = "right",
#     legend.title = element_blank(),
#     plot.title = element_text(hjust = 0.5)
#   ) +
#   # scale_color_manual(values=c(rev(scales::hue_pal()(3))))
#   guides(colour = guide_legend(reverse = T))
# 
# perf_plot

# save the plot if needed
# ggsave(
#   filename = paste0("lm_rf_h", hidden, "_", "r", R, "_plot.png"),
#   path = here::here("plots"),
#   plot = perf_plot,
#   width = 12.5,
#   height = 9,
#   dpi = 500
# )

# ------------------------------------------------------------------------------
# train a random forest and linear regression on the original data for comparison
set.seed(1)
randomForest <- caret::train(Y ~ ., df_train, method = "rf", trControl = trainControl(method = "none"))
lm_mod_train <- lm(Y ~ ., data = df_train)

#-------------------------------------------------------------------------------
# measuring the performance on the training and test sets
# create the variables needed for the model performances
model_list <- list(lm_mod_train, NA, randomForest) # we can replace our approach with an NA since it's a function
model_names <- c("simple_lm", paste0("em_lm_lm_", "h", hidden, "_r", R), "simple_rf")
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
    if (grepl("em_lm_lm_", model_names[j], fixed = TRUE)) {
      mod_pred <- infer_mlinear_lm(sub)
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
# postResample(pred = infer_mlinear_lm(df_train), obs = y_train)
# postResample(pred = predict(randomForest, x_train), obs = y_train)
#
# # validation set
# postResample(pred = predict(lm_mod_train, newdata = df_valid), obs = y_valid)
# postResample(pred = infer_mlinear_lm(df_valid), obs = y_valid)
# postResample(pred = predict(randomForest, x_valid), obs = y_valid)
#
# # test set
# postResample(pred = predict(lm_mod_train, newdata = df_test),obs = y_test)
# postResample(pred = infer_mlinear_lm(df_test), obs = y_test)
# postResample(pred = predict(randomForest, x_test), obs = y_test)