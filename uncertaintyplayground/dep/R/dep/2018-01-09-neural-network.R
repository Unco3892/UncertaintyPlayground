## ----spirals, echo = FALSE-------------------------------------------------------
# own data
x <- matrix(rnorm(4 * 100), nrow = 100)
y <- x[, 1] + x[, 2] + x[, 3] + x[, 4] + rnorm(100)
train_df <- data.frame(x, y)

## ----logistic-regression, echo = 1-----------------------------------------------
lm_reg <- lm(y ~ ., data = train_df)
summary(lm_reg)

## ----feedforward-----------------------------------------------------------------
feedforward <- function(x, w1, w2) {
  z1 <- cbind(1, x) %*% w1
  h <- z1
  z2 <- cbind(1, h) %*% w2
  list(output = z2, h = h)
}

## ----backprop--------------------------------------------------------------------
backpropagate <- function(x, y, y_hat, w1, w2, h, learn_rate) {
  dw2 <- t(cbind(1, h)) %*% (y_hat - y)
  dh <- (y_hat - y) %*% t(w2[-1, , drop = FALSE])
  dw1 <- t(cbind(1, x)) %*% (dh)

  w1 <- w1 - learn_rate * dw1
  w2 <- w2 - learn_rate * dw2

  list(w1 = w1, w2 = w2)
}

## ----train-----------------------------------------------------------------------
train <- function(x, y, hidden = 5, learn_rate = 1e-3, iterations = 1e4) {
  # note that if the learning rate is too small, you will have exploding gradients
  d <- ncol(x) + 1
  w1 <- matrix(rnorm(d * hidden), d, hidden)
  w2 <- as.matrix(rnorm(hidden + 1))
  for (i in 1:iterations) {
    ff <- feedforward(x, w1, w2)
    bp <- backpropagate(x, y,
      y_hat = ff$output,
      w1, w2,
      h = ff$h,
      learn_rate = learn_rate
    )
    w1 <- bp$w1
    w2 <- bp$w2
  }
  list(output = ff$output, w1 = w1, w2 = w2)
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
  w1 = nnet5$w1,
  w2 = nnet5$w2
)

ff_grid$output

## ----r6--------------------------------------------------------------------------
# library(R6)
# NeuralNetwork <- R6Class("NeuralNetwork",
#   public = list(
#     X = NULL,  Y = NULL,
#     W1 = NULL, W2 = NULL,
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
#       self$W1 <- .01 * matrix(rnorm(D * H), D, H)
#       self$W2 <- .01 * matrix(rnorm((H + 1) * K), H + 1, K)
#     },
#     fit = function(data = self$X) {
#       h <- self$sigmoid(data %*% self$W1)
#       score <- cbind(1, h) %*% self$W2
#       return(self$softmax(score))
#     },
#     feedforward = function(data = self$X) {
#       self$output <- self$fit(data)
#       invisible(self)
#     },
#     backpropagate = function(lr = 1e-2) {
#       h <- self$sigmoid(self$X %*% self$W1)
#       Yid <- match(self$Y, sort(unique(self$Y)))
#
#       haty_y <- self$output - (col(self$output) == Yid) # E[y] - y
#       dW2 <- t(cbind(1, h)) %*% haty_y
#
#       dh <- haty_y %*% t(self$W2[-1, , drop = FALSE])
#       dW1 <- t(self$X) %*% (self$dsigmoid(h) * dh)
#
#       self$W1 <- self$W1 - lr * dW1
#       self$W2 <- self$W2 - lr * dW2
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
