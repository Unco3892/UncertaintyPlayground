X <- matrix(rnorm(4 * 100), nrow = 100)
Y <- as.matrix(X[, 1] + X[, 2] + X[, 3] + X[, 4] + rnorm(100))
train_df <- data.frame(X, Y)


N <- length(y) ## also dim(X)[1]
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

continue <- TRUE
while (continue) {
  # random generation of weights
  theta_ws <- cbind(1, matrix(rnorm(4 * 1), 1, 4))
  # we assume an intercept of 1 here
  z_t <- theta_ws %*% t(cbind(1, X))

  # ILIA COMMENT: How come `modelL_p` is not defined here?
  ## Compute the z_t -> forward pass through modL_p
  z_t <- predict(modL_p, newdata = X) ## N x 1

  ## simulate the z_tilde -> use modL_p
  for (r in 1:R) {
    z_R[, r] <- rnorm(n = N, mean = z_t, sd = 1) ## z_R is N x R
  }

  ## Compute the y_tilde -> forward pass through modR_p
  for (r in 1:R) {
    y_R[, r] <- predict(modR_p, newdata = z_R[, r]) ## y_R (N x R)
  }

  ## Compute the weights -> apply the density of modR_p
  for (r in 1:R) {
    w_R[, r] <- dnorm(y, mean = y_R[, r], sd = 1)
  }
  w_sumbyrow <- apply(w_R, 1, sum) ## N x 1
  w_R <- sweep(w_R, 1, w_sumbyrow, FUN = "/") ## standardized w_R by their row-wise sums

  ## unfold w_R from (N x R) to (NR x 1) vector
  w_comp <- c(w_R)

  ## unfold z_R from (N x R) to (NR x 1) vector
  zR_comp <- c(z_R)

  ## optimize Q-function for the left model
  modL <- lm(zR_comp ~ X_comp, weights = w_comp)
  modL_p <- modL

  ## optimize Q-function for the right model
  modR <- lm(y_comp ~ zR_comp, weights = w_comp)
  modR_p <- modR

  if (converged) {
    continue <- FALSE
  }
}
