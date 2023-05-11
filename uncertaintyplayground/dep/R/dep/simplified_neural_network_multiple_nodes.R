library(tidyverse)
library(caret)

# general parameters
set.seed(1)
N <- 200
smp_size <- floor(0.5 * N)
train_ind <- sample(seq_len(N), size = smp_size)
R <- 500

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
df_test <- df[-train_ind, ]
x_train <- select(df_train, -Y) %>% as.matrix()
y_train <- df_train$Y %>% as.matrix()
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

relu <- function(x) max(0, x)
softplus <- function(x) log(1 + exp(x))

# iteration counter
i <- 0
continue <- TRUE
converged <- 10

while (continue) {
  # inside the while loop for the iteration
  i <- i + 1

  if (i == 1) {
    # get the dimensions
    set.seed(10)
    d <- ncol(x_train) + 1
    hidden <- 10

    # define the `z_R` array
    z_R <- array(NA, c(smp_size, hidden, R))

    # define left model and generate the weights based on the number of nodes
    modL_p <- matrix(rnorm(d * hidden), d, hidden)
  } else {
    # the issue is the weight's parameter in `lm` if using `predict()` -> to be discussed with Marc-O
    modL_p <-
      do.call(cbind, sapply(modL_p, function(g) {
        g[c("coefficients")]
      }))
  }
  # make prediction with the left model
  z_t <- cbind(1, x_train) %*% modL_p

  # we can apply an activation function to it
  # work on this -> applying it after the simulation
  # z_t <- apply(z_t, 2, relu)
  # z_t <- softplus(z_t)

  ## simulate the z_tilde -> use modL_p
  for (l in 1:hidden) {
    for (r in 1:R) {
      z_R[, l, r] <-
        rnorm(
          n = smp_size,
          mean = z_t[, l],
          sd = 1
        ) ## z_R is N x L x R
    }
  }

  # z_t <- relu(z_t)

  # we move to the right model
  if (i == 1) {
    # define the right model
    modR_p <- as.matrix(rnorm(hidden + 1))
  } else {
    # the issue is the weight's parameter in `lm` if using `predict()` -> to be discussed with Marc-O
    modR_p <- modR_p$coefficients %>% matrix()
  }

  ## Generate the y_tilde
  y_R <- apply(z_R, 3, function(p) {
    cbind(1, p) %*% modR_p
  }) # R is the 3rd dimension of `z_R`, ## y_R (N x R)

  ## Compute the weights -> apply the density of modR_p
  w_R <- apply(y_R, 2, function(y_estimate) {
    dnorm(y_train, mean = y_estimate, sd = 1)
  }) # R is the 2nd dimension

  ### just not sure about my way of doing `dnrom` vs yours
  denominator <- apply(w_R, 1, sum)

  ### operation sweep does the same
  w_R <- w_R / denominator ## standardized w_R by their row-wise sums
  # w_R <-sweep(w_R, 1, w_sumbyrow, FUN = "/") ## standardized w_R by their row-wise sums

  ## unfold w_R from (N x R) to (NR x 1) vector
  w_comp <- c(w_R)

  # how would this lm work? Since the weights are z_r's are different). This means a linear regression for each
  # we have to define multiple z models
  # this process can be parallalized

  ## optimize Q-function for the left model
  modLs <- list()
  for (l in 1:hidden) {
    ## unfold w_R from (N x R) to (NR x 1) vector
    mod_1 <- c(z_R[, l, ])
    ## comput the lm model for that hidden node
    modLs[[l]] <- lm(mod_1 ~ x_comp, weights = w_comp)
  }
  # assign this to the current model
  modL_p <- modLs

  ## optimize Q-function for the right model
  hidden_node <- apply(z_R, 2, c)
  modR <- lm(y_comp ~ hidden_node, weights = w_comp)
  modR_p <- modR

  if (i == converged) {
    continue <- FALSE
  }
}

## print some summaries if needed for diagnostics
# summary(lm(Y ~ ., data = df_train))
# lapply(modL_p, summary)
# summary(modR_p)

# ------------------------------------------------------------------------------
# inference function
infer <- function(data,
                  mod_left = modL_p,
                  mod_right = modR_p) {
  # define the data set for testing
  x_test <- select(data, -Y) %>% as.matrix()
  y_test <- data$Y %>% as.matrix()

  # use the model from the left hand side
  modL_p_infer <-
    do.call(cbind, sapply(mod_left, function(g) {
      g[c("coefficients")]
    }))
  # make prediction with the left model
  z_t_infer <- cbind(1, x_test) %*% modL_p_infer

  ## simulate the z_tilde -> use mod_left
  set.seed(10)
  z_R_infer <- array(NA, c(smp_size, hidden, R))
  for (l in 1:hidden) {
    for (r in 1:R) {
      z_R_infer[, l, r] <-
        rnorm(
          n = smp_size,
          mean = z_t_infer[, l],
          sd = 1
        ) ## z_R_infer is N x R
    }
  }

  # use the right model
  modR_p_infer <- mod_right$coefficients %>% matrix()
  ## Generate the y_tilde
  y_R_infer <-
    apply(z_R_infer, 3, function(p) {
      cbind(1, p) %*% modR_p_infer
    }) # R is the 3rd dimension of `z_R`, ## y_R (N x R)

  return(rowMeans(y_R_infer))
}

# measure this performance on both the training and test sets
# training set
lm_mod_train <- lm(Y ~ ., data = df_train)
postResample(pred = predict(lm_mod_train), obs = y_train)
postResample(pred = infer(df_train), obs = y_train)


# test set
lm_mod_test <- lm(Y ~ ., data = df_test)
postResample(pred = predict(lm_mod_test), obs = y_test)
postResample(
  pred = predict(lm_mod_train, newdata = df_test),
  obs = y_test
)
postResample(pred = infer(df_test), obs = y_test)
