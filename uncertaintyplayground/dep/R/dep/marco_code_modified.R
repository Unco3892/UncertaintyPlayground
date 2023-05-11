library(tidyverse)

set.seed(1)
X <- matrix(rnorm(4 * 100), nrow = 100)
Y <- as.matrix(X[, 1] + X[, 2] + X[, 3] + X[, 4] + rnorm(100))
train_df <- data.frame(X, Y)


N <- length(Y) ## also dim(X)[1]
R <- 500

# z_R ## N x R
# w_R ## N x R
# y_R ## N x R

# modL_p ## left hand side model (p for prime)
# modR_p ## right hand side model (p for prime)

# ILIA QUESTION: `d` represents the number of input features?
X_comp <- NULL ## X replicated R times NR x d
y_comp <- NULL ## y replicated R times NR x 1
for (r in 1:R) {
  X_comp <- rbind(X_comp, X)
  y_comp <- rbind(y_comp, Y)
}

str(X_comp)
str(y_comp)


# iteration counter
i <- 0
continue <- TRUE
converged <- 10

while (continue) {
  # inside the while loop for the iteration
  i <- i + 1

  if (i == 1) {
    # define the variables
    z_R <- matrix(NA, N, R)
    y_R <- matrix(NA, N, R)
    w_R <- matrix(NA, N, R)

    # define left model
    set.seed(1)
    d <- ncol(X) + 1
    hidden <- 1
    modL_p <- matrix(rnorm(d * hidden), d, hidden)
  } else {
    # the issue is the weight's parameter in `lm` if using `predict()` -> to be discussed with Marc-O
    modL_p <- modL_p$coefficients %>% matrix()
  }
  # make prediction with the left model
  z_t <- cbind(1, X) %*% modL_p

  ## simulate the z_tilde -> use modL_p
  for (r in 1:R) {
    z_R[, r] <- rnorm(n = N, mean = z_t, sd = 1) ## z_R is N x R
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
    w_R[, r] <- dnorm(Y, mean = y_R[, r], sd = 1)
  }
  w_sumbyrow <- apply(w_R, 1, sum) ## N x 1
  w_R <- sweep(w_R, 1, w_sumbyrow, FUN = "/") ## standardized w_R by their row-wise sums

  ## unfold w_R from (N x R) to (NR x 1) vector
  w_comp <- c(w_R)

  ## unfold z_R from (N x R) to (NR x 1) vector
  zR_comp <- c(z_R)

  ## optimize Q-function for the left model

  # is this message showing a problem?
  ## is this right? Warning message:
  ###  'newdata' had 100 rows but variables found have 50000 rows
  modL <- lm(zR_comp ~ X_comp, weights = w_comp)
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

summary(modL_p)
summary(modR_p)

# can even cut the code but altering the coefficients as described here
# https://stackoverflow.com/questions/25695565/predict-with-arbitrary-coefficients-in-r
