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
X <- matrix(rnorm(4 * N), nrow = N)
Y <- as.matrix(X[, 1] + X[, 2] + X[, 3] + X[, 4] + rnorm(N))
df <- data.frame(X, Y)

# ------------------------------------------------------------------------------
# working on the SRED (real) data
# df <- read_csv(here::here("data/metadata/train_data.csv"))
# df <- scale(df[, c(2:5)]) %>% bind_cols(Y = log(df$price))
# df <-  df[complete.cases(df),] %>% .[sample(nrow(.),size = N),]
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

# test against other models and see the performance
summary(lm(Y ~ ., data = df_train))
summary(lm(Y ~ ., data = df_test))

# N <- length(Y) ## also dim(X)[1]


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
# softplus <- function(x) log(1+exp(x))
# relu <- function(x) sapply(x, function(z) max(0,z))

# iteration counter
i <- 0
continue <- TRUE
converged <- 100

while (continue) {
  # inside the while loop for the iteration
  i <- i + 1

  if (i == 1) {
    # define the variables
    z_R <- matrix(NA, smp_size, R)
    y_R <- matrix(NA, smp_size, R)
    w_R <- matrix(NA, smp_size, R)

    # define left model
    set.seed(1)
    d <- ncol(x_train) + 1
    hidden <- 1
    modL_p <- matrix(rnorm(d * hidden), d, hidden)
  } else {
    # the issue is the weight's parameter in `lm` if using `predict()` -> to be discussed with Marc-O
    modL_p <- modL_p$coefficients %>% matrix()
  }
  # make prediction with the left model
  z_t <- cbind(1, x_train) %*% modL_p

  ## simulate the z_tilde -> use modL_p
  for (r in 1:R) {
    z_R[, r] <- rnorm(n = smp_size, mean = z_t, sd = 1) ## z_R is N x R
  }

  # we move to the right model
  if (i == 1) {
    # define the right model
    modR_p <- as.matrix(rnorm(hidden + 1))
  } else {
    # the issue is the weight's parameter in `lm` if using `predict()` -> to be discussed with Marc-O
    modR_p <- modR_p$coefficients %>% matrix()
  }
  for (r in 1:R) {
    y_R[, r] <- cbind(1, z_R[, r]) %*% modR_p ## y_R (N x R)
  }
  ## Compute the weights -> apply the density of modR_p
  for (r in 1:R) {
    w_R[, r] <- dnorm(y_train, mean = y_R[, r], sd = 1)
  }
  w_sumbyrow <- apply(w_R, 1, sum) ## N x 1
  # THIS PRODUCES ALL SAME NUMBERS
  w_R <- sweep(w_R, 1, w_sumbyrow, FUN = "/") ## standardized w_R by their row-wise sums

  ## unfold w_R from (N x R) to (NR x 1) vector
  w_comp <- c(w_R)

  ## unfold z_R from (N x R) to (NR x 1) vector
  zR_comp <- c(z_R)

  ## optimize Q-function for the left model

  # is this message showing a problem?
  ## is this right? Warning message:
  ###  'newdata' had 100 rows but variables found have 50000 rows
  modL <- lm(zR_comp ~ x_comp, weights = w_comp)
  modL_p <- modL

  ## optimize Q-function for the right model
  modR <- lm(y_comp ~ zR_comp, weights = w_comp)
  modR_p <- modR

  if (i == converged) {
    continue <- FALSE
  }
  # if (converged) {
  #   continue <- FALSE
  # }
}

# inference function
infer <- function(data, mod_left = modL_p, mod_right = modR_p) {
  # define the data set for testing
  x_test <- select(data, -Y) %>% as.matrix()
  y_test <- data$Y %>% as.matrix()

  # use the model from the left hand side
  modL_p_infer <- mod_left$coefficients %>% matrix()
  # make prediction with the left model
  z_t_infer <- cbind(1, x_test) %*% modL_p_infer

  ## simulate the z_tilde -> use modL_p
  set.seed(10)
  z_R_infer <- matrix(NA, smp_size, R)
  for (r in 1:R) {
    z_R_infer[, r] <- rnorm(n = smp_size, mean = z_t_infer, sd = 1)
  }
  # we use the right model
  modR_p_infer <- mod_right$coefficients %>% matrix()
  y_R_infer <- matrix(NA, smp_size, R)
  for (r in 1:R) {
    y_R_infer[, r] <-
      cbind(1, z_R_infer[, r]) %*% modR_p_infer ## y_R (N x R)
  }
  # return (y_R_infer)
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
