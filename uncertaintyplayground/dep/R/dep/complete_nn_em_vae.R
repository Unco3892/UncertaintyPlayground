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

## --we set the training parameters
epochs <- 1
batch_size <- 5
sigma_squared_primes <- nrow(train_df) / batch_size

## ----linear-regression--------------------------------------------------------
lm_reg <- lm(y ~ ., data = train_df)
summary(lm_reg)

## ----feedforward-----------------------------------------------------------------
# define the feedforward function
feedforward <- function(x, wts) {
  # initialize the activations for the input layer to the input data
  activations <- list(x)

  # loop over the wts and apply the feedforward calculation layer by layer
  for (i in 1:(length(wts) - 1)) {
    # check that the dimensions of the activations and wts are compatible for matrix multiplication
    if (ncol(activations[[i]]) != nrow(wts[[i]])) {
      stop("Incompatible dimensions for matrix multiplication in feedforward calculation")
    }
    z <- activations[[i]] %*% wts[[i]]
    # apply the activation function to the weighted sum
    h <- relu(z)
    # add the activations of the current layer to the list
    activations <- c(activations, list(h))
  }

  # the output of the feedforward calculation is the activations of the final layer
  output <- activations[[length(wts)]]

  # return the output and the activations at each layer
  list(output = output, activations = activations)
}

# define the backpropagate function
backpropagate <- function(x, y, y_hat, wts, activations, learn_rate) {
  # initialize the list of weight gradients to empty lists
  weight_grads <- rep(list(NULL), length(wts))

  # compute the gradient for the final layer
  weight_grads[[length(wts)]] <- t(cbind(1, activations[[length(wts) - 1]])) %*% (y_hat - y)

  # loop over the layers in reverse order
  for (i in (length(wts) - 1):2) {
    # compute the gradient for the current layer
    dh <- (y_hat - y) %*% t(wts[[i]][-1, , drop = FALSE])
    weight_grads[[i - 1]] <- t(cbind(1, activations[[i - 1]])) %*% (dh)
  }

  # update the wts
  for (i in 1:length(wts)) {
    wts[[i]] <- wts[[i]] - learn_rate * weight_grads[[i]]
  }

  # return the updated wts
  wts
}

## ----train-----------------------------------------------------------------------
train <- function(x, y, layer_sizes, learn_rate = 1e-3, iterations = 1e4) {
  # note that if the learning rate is too small, you will have exploding gradients
  d <- ncol(x)
  # initialize the wts using the list of layer sizes
  wts_1 <- lapply(layer_sizes[-length(layer_sizes)], function(size) {
    matrix(rnorm(d * size), d, size)
  })
  # add an extra weight matrix for the output layer
  wts_2 <- c(wts, list(as.matrix(rnorm(layer_sizes[length(layer_sizes)] + 1))))
  # return (list(wts_1))

  # return (list(x, wts_1, wts_2))

  for (i in 1:iterations) {
    ff <- feedforward(x, wts)
    # bp <- backpropagate(x, y,
    #                     y_hat = ff$output,
    #                     wts,
    #                     activations = ff$activations,
    #                     learn_rate = learn_rate)
    # wts <- bp
  }
  list(output = ff$output, wts = wts)
}

x <- data.matrix(train_df[, c("X1", "X2", "X3", "X4")])
y <- train_df$y
nnet_2layers <- train(x, y, layer_sizes = c(4, 10, 1), iterations = 1e5)
str(nnet_2layers)


## ----accuracy-ad-hoc-------------------------------------------------------------
mean((nnet5$output - y)^2)
