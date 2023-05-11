library(tidyverse)
library(caret)
library(broom)
library(nnet)
library(parallel)
library(doParallel)

# general parameters
set.seed(1)
N <- 800
R <- 100
train_size <- 0.5
valid_size <- 0.25
train_smp_size <- floor(train_size * N)
train_ind <- sample(seq_len(N), size = train_smp_size)
valid_smp_size <- floor(valid_size * N)
valid_ind <- sample(seq_len(train_smp_size), size = valid_smp_size)
train_smp_size <- train_smp_size - valid_smp_size

#-------------------------------------------------------------------------------
# working with simulated data
# X <- matrix(rnorm(4 * N), nrow = N)
# Y <- as.matrix(X[, 1] + X[, 2] + X[, 3] + X[, 4] + rnorm(N))
# df <- data.frame(X, Y)

# ------------------------------------------------------------------------------
# working on the SRED (real) data
df <- read_csv(here::here("data/metadata/train_data.csv"))
df <- scale(df[, c(2:5)]) %>% bind_cols(Y = log(df$price))
df <- df[complete.cases(df), ] %>% .[sample(nrow(.), size = N), ]
# dim(y_train) <- c(length(y_train), 1) # add extra dimension to vector

# ------------------------------------------------------------------------------
# divide between train and test
df_train <- df[train_ind, ]
df_valid <- df_train[valid_ind, ]
df_train <- df_train[-valid_ind, ]
df_test <- df[-train_ind, ]

x_train <- select(df_train, -Y) %>% as.matrix()
y_train <- df_train$Y %>% as.matrix()

x_valid <- select(df_valid, -Y) %>% as.matrix()
y_valid <- df_valid$Y %>% as.matrix()

x_test <- select(df_test, -Y) %>% as.matrix()
y_test <- df_test$Y %>% as.matrix()

# ------------------------------------------------------------------------------

## variable definitions
### z_R ## N x R
### w_R ## N x R
### y_R ## N x R
### modL_p ## left hand side model (p for prime)
### modR_p ## right hand side model (p for prime)


# ILIA QUESTION: `d` represents the number of input features?
x_comp <- NULL ## X replicated R times NR x d
y_comp <- NULL ## y replicated R times NR x 1
for (r in 1:R) {
  x_comp <- rbind(x_comp, x_train)
  y_comp <- rbind(y_comp, y_train)
}

# custom activation functions
# relu <- function(x)
#   sapply(x, function(z)
#     max(0, z))

relu <- function(x) {
  ifelse(x > 0, x, 0)
}
softplus <- function(x) {
  log(1 + exp(x))
}

# iteration counter
i <- 0
continue <- TRUE
converged <- 1000

modelL_perf <- list()
modelR_perf <- list()
train_perf <- list()
valid_perf <- list()

# parallelization options
# cl <- makePSOCKcluster(detectCores() - 2)
# registerDoParallel(cl)

# myControl <- trainControl(## 3-fold CV
#   method = "cv",
#   number = 6)

nnGrid <- expand.grid(
  size = seq(1, 12, 4),
  decay = c(0, 0.2, 0.4)
)


# ------------------------------------------------------------------------------
# inference function
infer <- function(data,
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
  # return (rowMeans(y_R_infer))
  # can also calculate the median
  return(apply(y_R_infer, 1, median))
}
#-------------------------------------------------------------------------------
start_time <- Sys.time()
while (continue) {
  # inside the while loop for the iteration
  i <- i + 1

  if (i == 1) {
    # get the dimensions
    set.seed(10)
    d <- ncol(x_train) + 1
    hidden <- 5

    # define the `z_R` array
    z_R <- array(NA, c(train_smp_size, hidden, R))

    # define left model and generate the weights based on the number of nodes
    modL_p <- matrix(rnorm(d * hidden), d, hidden)
    # make prediction with the left model
    z_t <- cbind(1, x_train) %*% modL_p
  } else {
    # make prediction with the left model
    z_t <- do.call(cbind, lapply(modL_p, function(g) {
      predict(g, x_train)
    }))
  }

  # we can apply an activation function to it
  # work on this -> applying it after the simulation
  # z_t <- softplus(z_t)
  # z_t <- relu(z_t)

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

  # z_R <- relu(z_R)
  # z_t <- softplus(z_t)

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
      predict(modR_p, data.frame(p))
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
  # w_R <-sweep(w_R, 1, w_sumbyrow, FUN = "/") ## standardized w_R by their row-wise sums

  ## unfold w_R from (N x R) to (NR x 1) vector
  w_comp <- c(w_R)

  # how would this lm work? Since the weights are z_r's are different). This means a linear regression for each
  # we have to define multiple z models
  # this process can be parallalized

  # define the neural network parameters

  ## optimize Q-function for the left model
  modLs <- list()
  for (l in 1:hidden) {
    ## unfold w_R from (N x R) to (NR x 1) vector
    mod_1 <- c(z_R[, l, ])
    ## comput the lm model for that hidden node
    set.seed(1234)
    modLs[[l]] <- train(
      mod_1 ~ .,
      data = data.frame(x_comp, mod_1),
      method = "lm",
      # maxit = 1000,
      # linout = T,
      weights = w_comp,
      # tuneGrid = nnGrid,
      # trControl = myControl
      trControl = trainControl(method = "none")
    )
  }

  # assign this to the current model
  modL_p <- modLs

  # modelL_perf[[i]] <- bind_rows(lapply(modL_p, function(x)
  #   glance(x) %>%
  #     select(1:2) %>% mutate(run = i)))

  ## optimize Q-function for the right model
  hidden_node <- apply(z_R, 2, c)
  # modR <- lm(y_comp ~ hidden_node , weights = w_comp)
  # modR <- nnet(hidden_node, y_comp, size = 5, weights = w_comp, linout = T,maxit = 1000)

  modR <- train(
    y_comp ~ .,
    data = data.frame(hidden_node, y_comp),
    method = "lm",
    # maxit = 1000,
    # linout = T,
    weights = w_comp,
    # tuneGrid = nnGrid,
    # trControl = myControl
    trControl = trainControl(method = "none")
  )

  modR_p <- modR

  # log the performance metric of the right side
  modelR_perf[[i]] <- modR_p

  # log the overall perpformance metrics (training and validation)
  train_perf[[i]] <-
    postResample(pred = infer(df_train), obs = y_train)
  valid_perf[[i]] <-
    postResample(pred = infer(df_valid, size = valid_smp_size), obs = y_valid)

  # printing diagnostics
  message(
    "Iteration: ",
    i,
    ", Train R2 & MAE: ",
    round(train_perf[[i]][2], 3),
    " & ",
    round(train_perf[[i]][3], 3),
    ", Validation R2 & MAE: ",
    round(valid_perf[[i]][2], 3),
    " & ",
    round(valid_perf[[i]][3], 3)
  )

  if (i == converged) {
    continue <- FALSE
  }
}

end_time <- Sys.time()
end_time - start_time

## When you are done:
# stopCluster(cl)


# unregister_dopar <- function() {
#   env <- foreach:::.foreachGlobals
#   rm(list = ls(name = env), pos = env)
# }
#
# unregister_dopar()

# ------------------------------------------------------------------------------
# plot the performance
bind_rows(lapply(modelR_perf, function(x) {
  glance(summary(x)) %>%
    select(1:2)
})) %>%
  View()

# bind_rows(lapply(modelR_perf, function(x)
#   x$results %>%
#     select(2:4))) %>%
#   tibble()


# ------------------------------------------------------------------------------

# measure this performance on both the training and test sets
# training set
lm_mod_train <- lm(Y ~ ., data = df_train)
postResample(pred = predict(lm_mod_train), obs = y_train)
# postResample(pred = predict(lm_mod_train,newdata = df_train), obs = y_train)
postResample(pred = infer(df_train), obs = y_train)


# test set
# lm_mod_test <- lm(Y ~ ., data = df_test)
# postResample(pred = predict(lm_mod_test), obs = y_test)
postResample(
  pred = predict(lm_mod_train, newdata = df_test),
  obs = y_test
)
postResample(pred = infer(df_test, size = nrow(df_test)), obs = y_test)
# postResample(pred = infer(df_valid,size = nrow(df_valid)), obs = y_valid)

# ------------------------------------------------------------------------------
library(neuralnet)
f <-
  as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))

# %>% str()

# nn <- neuralnet(y_comp ~ . ,
#                 data= data.frame(y_comp, hidden_node),
#                 hidden=c(5),
#                 rep = 10,
#                 stepmax = 1000,
#                 threshold = 5,
#                 lifesign = "minimal",
#                 linear.output=T,)
#
#
# nn.results <- neuralnet::compute(nn, data.frame(hidden_node),rep = 3)
#
# # postResample(pred = predict(lm_mod_train,newdata = df_train), obs = y_train)
#
#
# nn <- nnet(y_comp, y_comp, size = 5, linout = T)
#
# nn <- nnet(hidden_node, y_comp, size = 5, weights = w_comp, linout = T,maxit = 1000)
# nn <- nnet(hidden_node, y_comp, size = 5, linout = T,maxit = 1000)
#
# postResample(pred = predict(modR_p, newdata = data.frame(hidden_node)), obs = y_comp)
# postResample(pred = predict(nn, newdata = data.frame(hidden_node)), obs = y_comp)
#
#
# nn <- nnet(x_train, y_train, size = 5, linout = T,maxit = 100000)
#
# nn
#
# postResample(pred = predict(nn, newdata = x_train), obs = y_train)
# postResample(pred = predict(nn, newdata = x_test), obs = y_test)


# library(mclust)
# mixclust <- Mclust(df_train)
# plot(mixclust, what=c("uncertainty"))  # plot the distinct clusters found
#
#
# # diagnostics
# f <- bind_rows(modelL_perf, .id = "run")
#
# ggplot(f, aes(run, r.squared, colour = factor(run))) +
#   geom_point()
#
# bind_rows(lapply(modelR_perf, function(x)
#   glance(x) %>%
#     select(1:2)))
#
# glance(modR_p) %>%
#   select(1:2)
#
# data %>%
#   group_by(a) %>%
#   do(tidy(lm(b ~ c, data = .))) %>%
#   select(variable = a, t_stat = statistic) %>%
#   slice(2)
#
#
# lapply(lapply(modLs, summary), function(x) x$adj.r.squared)
# lapply(lapply(modelR_perf, summary), function(x) x$adj.r.squared)
#
#
# dfs <-
#
# bind_rows(lapply(modelL_perf, data.frame, stringsAsFactors = FALSE))
#

## print some summaries if needed for diagnostics
# summary(lm(Y ~ ., data = df_train))
# lapply(modL_p, summary)
# summary(modR_p)
