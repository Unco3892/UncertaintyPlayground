# define inference function
# ------------------------------------------------------------------------------
## inference function for two linear models
infer_lm_lm <- function(data,
                        size = train_smp_size,
                        mod_left = modL_p,
                        mod_right = modR_p) {
  # define the data set for testing
  final_x_test <- select(data, -Y) %>% as.matrix()

  # make prediction with the left model
  z_t_infer <- do.call(cbind, lapply(mod_left, function(g) {
    predict(g, final_x_test)
  }))

  # return (list(final_x_test, z_t_infer))
  ## simulate the z_tilde -> use mod_left
  set.seed(10)


  z_R_infer <- array(NA, c(size, hidden, R))
  for (l in 1:hidden) {
    for (r in 1:R) {
      z_R_infer[, l, r] <-
        rnorm(
          n = size,
          mean = z_t_infer[, l],
          sd = 1
        ) ## z_R_infer is N x R
    }
  }

  ## Generate the y_tilde
  y_R_infer <- apply(z_R_infer, 3, function(p) {
    predict(mod_right, data.frame(p))
  })

  # return(y_R_infer)
  return(rowMeans(y_R_infer))
  # can also calculate the median
  # return (apply(y_R_infer, 1, median))
}


## inference function for two neural networks (left and right model)
infer_nn_nn <- function(data,
                        size = train_smp_size,
                        mod_left = modL_p,
                        mod_right = modR_p) {
  # define the data set for testing
  final_x_test <-
    select(data, -Y) %>% as.matrix()

  # make prediction with the left model
  z_t_infer <-
    predict(mod_left, final_x_test, verbose = 0, use_multiprocessing = T)

  # return (list(final_x_test, z_t_infer))
  ## simulate the z_tilde -> use mod_left
  set.seed(10)


  z_R_infer <- array(NA, c(size, hidden, R))
  for (l in 1:hidden) {
    for (r in 1:R) {
      z_R_infer[, l, r] <-
        rnorm(
          n = size,
          mean = z_t_infer[, l],
          sd = 1
        ) ## z_R_infer is N x R
    }
  }

  ## Generate the y_tilde
  y_R_infer <- apply(z_R_infer, 3, function(p) {
    predict(mod_right, p, verbose = 0, use_multiprocessing = T)
  })

  # return(y_R_infer)
  return(rowMeans(y_R_infer))
  # can also calculate the median
  # return (apply(y_R_infer, 1, median))
}

# inference function for neural network and caret train (left and right model)
infer_nn_caret <- function(data,
                           size = train_smp_size,
                           mod_left = modL_p,
                           mod_right = modR_p) {
  # define the data set for testing
  final_x_test <- select(data, -Y) %>% as.matrix()

  # make prediction with the left model
  z_t_infer <-
    predict(mod_left, final_x_test, verbose = 0, use_multiprocessing = T)

  # return (list(final_x_test, z_t_infer))
  ## simulate the z_tilde -> use mod_left
  set.seed(10)

  z_R_infer <- array(NA, c(size, hidden, R))
  for (l in 1:hidden) {
    for (r in 1:R) {
      z_R_infer[, l, r] <-
        rnorm(
          n = size,
          mean = z_t_infer[, l],
          sd = 1
        ) ## z_R_infer is N x R
    }
  }

  ## Generate the y_tilde
  y_R_infer <- apply(z_R_infer, 3, function(p) {
    predict(mod_right, data.frame(p))
  })

  # return(y_R_infer)
  return(rowMeans(y_R_infer))
  # can also calculate the median
  # return (apply(y_R_infer, 1, median))
}

# inference function for multi-variate linear model and caret random forest (left and right models)
infer_multivlm_caret <- function(data,
                                 size = train_smp_size,
                                 mod_left = modL_p,
                                 mod_right = modR_p,
                                 final_sd = custom_sd) {
  # define the data set for testing
  final_x_test <- select(data, -Y)

  # make prediction with the left model
  z_t_infer <-
    predict(mod_left, final_x_test)

  # return (list(final_x_test, z_t_infer))
  ## simulate the z_tilde -> use mod_left
  set.seed(10)

  z_R_infer <- array(NA, c(size, hidden, R))
  for (l in 1:hidden) {
    for (r in 1:R) {
      z_R_infer[, l, r] <-
        rnorm(
          n = size,
          mean = z_t_infer[, l],
          sd = final_sd
        ) ## z_R_infer is N x R
    }
  }

  ## Generate the y_tilde
  y_R_infer <- apply(z_R_infer, 3, function(p) {
    predict(mod_right, data.frame(p))
  })

  # return(y_R_infer)
  return(rowMeans(y_R_infer))
  # can also calculate the median
  # return (apply(y_R_infer, 1, median))
}


# inference function for multi-variate random forest and caret random forest (left and right models)
infer_rf_rf <- function(data,
                        size = train_smp_size,
                        mod_left = modL_p,
                        mod_right = modR_p,
                        final_sd = custom_sd) {
  # define the data set for testing
  final_x_test <- select(data, -Y) %>% as.matrix()

  # make prediction with the left model
  z_t_infer <- predict(mod_left, data.frame(final_x_test))
  z_t_infer <- do.call(cbind, lapply(z_t_infer$regrOutput, "[[", "predicted"))

  # return (list(final_x_test, z_t_infer))
  ## simulate the z_tilde -> use mod_left
  set.seed(10)

  z_R_infer <- array(NA, c(size, hidden, R))
  for (l in 1:hidden) {
    for (r in 1:R) {
      z_R_infer[, l, r] <-
        rnorm(
          n = size,
          mean = z_t_infer[, l],
          sd = final_sd
        ) ## z_R_infer is N x R
    }
  }

  ## Generate the y_tilde
  y_R_infer <- apply(z_R_infer, 3, function(p) {
    predict(mod_right, data.frame(p))
  })

  # return(y_R_infer)
  return(rowMeans(y_R_infer))
  # can also calculate the median
  # return (apply(y_R_infer, 1, median))
}


# inference function for neural network and caret train (left and right model)
infer_multinn_caret <- function(test_tab,
                                test_img,
                                mod_left = modL_p,
                                mod_right = modR_p) {
  # make prediction with the left model
  z_t_infer <- predict(mod_left, list(test_tab, test_img), verbose = 0, use_multiprocessing = T)
  size <- nrow(test_tab)

  ## simulate the z_tilde -> use mod_left
  # define the `z_R` arrays (N x HIDDEN NODES FOR THAT MODALITY)
  z_R_tab_infer <- array(NA, c(size, hidden_tab, R))
  z_R_img_infer <- array(NA, c(size, hidden_img, R))

  set.seed(10)
  ## simulate the z_tilde for tabular data -> use modL_p_tab
  for (l in 1:hidden_tab) {
    for (r in 1:R) {
      z_R_tab_infer[, l, r] <- rnorm(
        n = size,
        mean = z_t_infer[, l],
        sd = 1
      ) ## z_R_tab is N x L x R
    }
  }

  ## simulate the z_tilde for image data -> use modL_p_img
  for (l in 1:hidden_img) {
    for (r in 1:R) {
      z_R_img_infer[, l, r] <- rnorm(
        n = size,
        mean = z_t_infer[, l + hidden_tab],
        sd = 1
      ) ## z_R_img is N x L x R
    }
  }

  # join all the hidden nodes together
  z_R_infer <- abind(list(z_R_tab_infer, z_R_img_infer), along = 2)

  ## Generate the y_tilde
  y_R_infer <- apply(z_R_infer, 3, function(p) {
    predict(
      mod_right, data.frame(p)
    )
  })

  return(rowMeans(y_R_infer))
  # can also calculate the median
  # return (apply(y_R_infer, 1, median))
}

# inference function for `rf` and the `XGboost`
infer_xgb_rf <- function(data,
                         size = train_smp_size,
                         mod_left = modL_p,
                         mod_right = modR_p,
                         final_sd = custom_sd) {
  # define the data set for testing
  final_x_test <- select(data, -Y) %>% as.matrix()

  # make prediction with the left model
  z_t_infer <- py_to_r(mod_left$predict(final_x_test))

  # return (list(final_x_test, z_t_infer))
  ## simulate the z_tilde -> use mod_left
  set.seed(10)

  z_R_infer <- array(NA, c(size, hidden, R))
  for (l in 1:hidden) {
    for (r in 1:R) {
      z_R_infer[, l, r] <-
        rnorm(
          n = size,
          mean = z_t_infer[, l],
          sd = final_sd
        ) ## z_R_infer is N x R
    }
  }

  ## Generate the y_tilde
  y_R_infer <- apply(z_R_infer, 3, function(p) {
    predict(mod_right, data.frame(p))
  })

  # return(y_R_infer)
  return(rowMeans(y_R_infer))
  # can also calculate the median
  # return (apply(y_R_infer, 1, median))
}


# inference function for `rf` and the `XGboost`
infer_xgb_xgb <- function(data,
                          size = train_smp_size,
                          mod_left = modL_p,
                          mod_right = modR_p,
                          final_sd = custom_sd) {
  # define the data set for testing
  final_x_test <- select(data, -Y) %>% as.matrix()

  # make prediction with the left model
  z_t_infer <- py_to_r(mod_left$predict(final_x_test))

  # return (list(final_x_test, z_t_infer))
  ## simulate the z_tilde -> use mod_left
  set.seed(10)

  z_R_infer <- array(NA, c(size, hidden, R))
  for (l in 1:hidden) {
    for (r in 1:R) {
      z_R_infer[, l, r] <-
        rnorm(
          n = size,
          mean = z_t_infer[, l],
          sd = final_sd
        ) ## z_R_infer is N x R
    }
  }

  ## Generate the y_tilde
  # y_R_infer <- apply(z_R_infer, 3, function(p) {
  #   predict(mod_right, data.frame(p))
  # })

  y_R_infer <- matrix(nrow = dim(z_R_infer)[1], ncol = dim(z_R_infer)[3])
  for (j in 1:dim(z_R_infer)[3]) {
    y_R_infer[, j] <- py_to_r(mod_right$predict(z_R_infer[, , j]))
  }

  # y_R_infer <- matrix(nrow=dim(z_R_infer)[1], ncol=dim(z_R_infer)[2])
  # for (i in 1:dim(z_R_infer)[3]) {
  #   y_R_infer[, , i] <- predict(mod_right, data.frame(z_R_infer[, , i]))
  # }

  # return(y_R_infer)
  return(rowMeans(y_R_infer))
  # can also calculate the median
  # return (apply(y_R_infer, 1, median))
}

# inference function for `rf` and the `XGboost`
infer_mlinear_lm <- function(data,
                             mod_left = modL_p,
                             mod_right = modR_p,
                             final_sd = custom_sd) {
  # define the data set for testing
  final_x_test <- select(data, -Y)

  # make prediction with the left model
  z_t_infer <- lapply(1:n_features, function(mm) {
    predict(mod_left[[mm]], newdata = final_x_test)
  })

  ## simulate the z_tilde -> use mod_left
  set.seed(10)

  # z_R_sep_infer <- foreach(p = 1:length(n_z_outcomes), .combine = 'list', .multicombine = T) %dopar% {
  # if (!is.matrix(z_t_infer[[p]])) {
  #   z_t_infer[[p]] <- as.matrix(z_t_infer[[p]])
  #   }
  #   sample_z_r(n_z_outcomes[p], z_means = z_t_infer[[p]], a_size = nrow(final_x_test))
  #   }

  z_R_sep_infer <- list()
  for (p in 1:length(n_z_outcomes)) {
    if (!is.matrix(z_t_infer[[p]])) {
      z_t_infer[[p]] <- as.matrix(z_t_infer[[p]])
    }
    z_R_sep_infer[[p]] <- sample_z_r(n_z_outcomes[p], z_means = z_t_infer[[p]], a_size = nrow(final_x_test))
  }

  # then we concatenate all the outcomes
  z_R_infer <- abind::abind(z_R_sep_infer, along = 2)
  ## Generate the y_tilde
  y_R_infer <- apply(z_R_infer, 3, function(p) {
    predict(mod_right, data.frame(p))
  })

	# Algorithm 10
  return(rowMeans(y_R_infer))
	# Algorithm 11
  # Find the index of the maximum prediction for each instance
  # max_pred_index <- apply(y_R_infer, 1, which.max)
  # Return the corresponding predictions
  # return(y_R_infer[cbind(seq_along(max_pred_index), max_pred_index)])
}


# inference function for the gaussian process regression
infer_multivlm_svgpr <- function(data,
                                 size = train_smp_size,
                                 mod_left = modL_p,
                                 mod_right = modR_p,
                                 final_sd = custom_sd) {
  # define the data set for testing
  final_x_test <- select(data, -Y)

  # make prediction with the left model
  z_t_infer <-
    predict(mod_left, final_x_test)

  # return (list(final_x_test, z_t_infer))
  ## simulate the z_tilde -> use mod_left
  set.seed(10)

  z_R_infer <- array(NA, c(size, hidden, R))
  for (l in 1:hidden) {
    for (r in 1:R) {
      z_R_infer[, l, r] <-
        rnorm(
          n = size,
          mean = z_t_infer[, l],
          sd = final_sd
        ) ## z_R_infer is N x R
    }
  }

  ## Generate the y_tilde with SVGPR
  y_R_infer <- apply(z_R_infer, 3, function(p) {
    mod_right$predict_with_uncertainty(p)[[1]]
  }) # R is the 3rd dimension of `z_R`, ## y_R (N x R)

  # return(y_R_infer)
  return(rowMeans(y_R_infer))
  # can also calculate the median
  # return (apply(y_R_infer, 1, median))
}
