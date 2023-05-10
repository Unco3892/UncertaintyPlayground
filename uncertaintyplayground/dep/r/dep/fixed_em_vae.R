# inspired from https://selbydavid.com/2018/01/09/neural-network/

# design the multi-modal approach in `R`
# let's apply this for a single modality which is tabular data
# afterwards, I can read the images and the tabular data and cast them into the
# same object

# load all the necessary libraries
library(tidyverse)

## -------data------------------------------------------------------------------
# generate/read the data
x <- matrix(rnorm(4 * 100), nrow = 100)
y <- x[, 1] + x[, 2] + x[, 3] + x[, 4] + rnorm(100)
train_df <- data.frame(x, y)

# train_data <-
#   read_csv(here::here("data/metadata/train_data.csv"))
# train <- head(train_data, 100)

## --for later: we set the training parameters
# epochs <- 1
# batch_size <- 5
# sigma_squared_primes <- nrow(train_df) / batch_size

## ----linear-regression--------------------------------------------------------
lm_reg <- lm(y ~ ., data = train_df)
summary(lm_reg)

## ----spirals, echo = FALSE-------------------------------------------------------
# own data
x <- matrix(rnorm(4 * 100), nrow = 100)
y <- x[, 1] + x[, 2] + x[, 3] + x[, 4] + rnorm(100)
train_df <- data.frame(x, y)

## ----logistic-regression, echo = 1-----------------------------------------------
lm_reg <- lm(y ~ ., data = train_df)
summary(lm_reg)

## ----feedforward-----------------------------------------------------------------
feedforward <- function(x, theta1, phi1, R, y_true) {
  # DOUBLE-CHECK THIS WITH THE LINK
  z_tilde <- cbind(1, x) %*% theta1
  h <- z_tilde
  # any other way than the for loop
  z <- replicate(R, {
    t(apply(h, 1, rnorm))
  })
  # y_tilde <- cbind(1, z) %*% phi1
  y_tilde <- apply(z, 3, function(x) {
    cbind(1, x) %*% phi1
  })
  # R is the 3rd dimension and not the 2nd one
  # symmetrical (which is the first arguemnt) so it does not change anything
  p_prime_y_z <- dnorm(y_true, y_tilde)
  # mapply(function(x, y){dnorm(x, y)}, y_tilde[,i], y_true[,i])
  denominator <- apply(p_prime_y_z, 1, sum)

  # operation sweep is clearer
  w_i_r <- p_prime_y_z / denominator

  return(
    list(
      x = x,
      y = y_true,
      theta_weights = theta1,
      z_tilde = h,
      z_simulated = z,
      y_tilde = y_tilde,
      pdf = p_prime_y_z,
      denominator = denominator,
      w_i_r = w_i_r
    )
  )
  list(output = z2, h_simulated = z_tilde)
}


# generate the weights
hidden <- 5
d <- ncol(x) + 1
theta1 <- matrix(rnorm(d * hidden), d, hidden)
phi1 <- as.matrix(rnorm(hidden + 1))

ff_grid <- feedforward(
  x = x,
  theta1 = theta1,
  phi1 = phi1,
  R = 500,
  y_true = y
)


modL <- lm(zR_comp ~ x_comp, weights = w_comp)

str(ff_grid)

str()


y_tilde <- apply(z, 2, function(x) {
  dnorm(x)
})




str(ff_grid)
str(ff_grid)

str(t(apply(ff_grid[[1]], 1, rnorm)))
dnorm()

head(ff_grid[[1]])
str(replicate(3, {
  apply(ff_grid[[1]], 1, rnorm)
}))


rnorm(c(1, 2, 3, 4, 5, 6))

## ----backprop--------------------------------------------------------------------
backpropagate <-
  function(x, y, y_hat, theta1, phi1, h, learn_rate) {
    dphi1 <- t(cbind(1, h)) %*% (y_hat - y)
    dh <- (y_hat - y) %*% t(phi1[-1, , drop = FALSE])
    dtheta1 <- t(cbind(1, x)) %*% (dh)

    theta1 <- theta1 - learn_rate * dtheta1
    phi1 <- phi1 - learn_rate * dphi1

    list(theta1 = theta1, phi1 = phi1)
  }

## ----train-----------------------------------------------------------------------
train <-
  function(x,
           y,
           hidden = 5,
           learn_rate = 1e-3,
           iterations = 1e4,
           R = 500) {
    # note that if the learning rate is too small, you will have exploding gradients
    d <- ncol(x) + 1
    theta1 <- matrix(rnorm(d * hidden), d, hidden)
    phi1 <- as.matrix(rnorm(hidden + 1))
    for (i in 1:iterations) {
      ff <- feedforward(x, theta1, phi1)
      bp <- backpropagate(
        x,
        y,
        y_hat = ff$output,
        theta1,
        phi1,
        h = ff$h_simulated,
        learn_rate = learn_rate
      )
      theta1 <- bp$theta1
      phi1 <- bp$phi1
    }
    list(
      output = ff$output,
      theta1 = theta1,
      phi1 = phi1
    )
  }

x <- data.matrix(train_df[, c("X1", "X2", "X3", "X4")])
y <- train_df$y
nnet5 <- train(x, y, hidden = 5, iterations = 1e5)

nnet5

## ----accuracy-ad-hoc-------------------------------------------------------------
mean((nnet5$output - y)^2)


## ----grid-ad-hoc-----------------------------------------------------------------
ff_grid <- feedforward(
  x = x,
  theta1 = nnet5$theta1,
  phi1 = nnet5$phi1
)

ff_grid$output

## ----r6--------------------------------------------------------------------------
# library(R6)
# NeuralNetwork <- R6Class("NeuralNetwork",
#   public = list(
#     X = NULL,  Y = NULL,
#     theta1 = NULL, phi1 = NULL,
#     output = NULL,
#     initialize = function(formula, hidden, data = list()) {
#       # Model and training data
#       mod <- model.frame(formula, data = data)
#       self$X <- model.matrix(attr(mod, 'terms'), data = mod)
#       self$Y <- model.response(mod)
#
#       # Dimensions
#       D <- ncol(self$X) # input dimensions (+ bias)
#       K <- length(unique(self$Y)) # number of classes
#       H <- hidden # number of hidden nodes (- bias)
#
#       # Initial weights and bias
#       self$theta1 <- .01 * matrix(rnorm(D * H), D, H)
#       self$phi1 <- .01 * matrix(rnorm((H + 1) * K), H + 1, K)
#     },
#     fit = function(data = self$X) {
#       h <- self$sigmoid(data %*% self$theta1)
#       score <- cbind(1, h) %*% self$phi1
#       return(self$softmax(score))
#     },
#     feedforward = function(data = self$X) {
#       self$output <- self$fit(data)
#       invisible(self)
#     },
#     backpropagate = function(lr = 1e-2) {
#       h <- self$sigmoid(self$X %*% self$theta1)
#       Yid <- match(self$Y, sort(unique(self$Y)))
#
#       haty_y <- self$output - (col(self$output) == Yid) # E[y] - y
#       dphi1 <- t(cbind(1, h)) %*% haty_y
#
#       dh <- haty_y %*% t(self$phi1[-1, , drop = FALSE])
#       dtheta1 <- t(self$X) %*% (self$dsigmoid(h) * dh)
#
#       self$theta1 <- self$theta1 - lr * dtheta1
#       self$phi1 <- self$phi1 - lr * dphi1
#
#       invisible(self)
#     },
#     predict = function(data = self$X) {
#       probs <- self$fit(data)
#       preds <- apply(probs, 1, which.max)
#       levels(self$Y)[preds]
#     },
#     compute_loss = function(probs = self$output) {
#       Yid <- match(self$Y, sort(unique(self$Y)))
#       correct_logprobs <- -log(probs[cbind(seq_along(Yid), Yid)])
#       sum(correct_logprobs)
#     },
#     train = function(iterations = 1e4,
#                      learn_rate = 1e-2,
#                      tolerance = .01,
#                      trace = 100) {
#       for (i in seq_len(iterations)) {
#         self$feedforward()$backpropagate(learn_rate)
#         if (trace > 0 && i %% trace == 0)
#           message('Iteration ', i, '\tLoss ', self$compute_loss(),
#                   '\tAccuracy ', self$accuracy())
#         if (self$compute_loss() < tolerance) break
#       }
#       invisible(self)
#     },
#     accuracy = function() {
#       predictions <- apply(self$output, 1, which.max)
#       predictions <- levels(self$Y)[predictions]
#       mean(predictions == self$Y)
#     },
#     sigmoid = function(x) 1 / (1 + exp(-x)),
#     dsigmoid = function(x) x * (1 - x),
#     softmax = function(x) exp(x) / rowSums(exp(x))
#   )
# )
#
#
# ## ----iris------------------------------------------------------------------------
# irisnet <- NeuralNetwork$new(Species ~ ., data = iris, hidden = 5)
#
#
# ## ----iris-train, collapse = TRUE-------------------------------------------------
# irisnet$train(9999, trace = 1e3, learn_rate = .0001)
#
#
# ## ----obj-------------------------------------------------------------------------
# irisnet
#
