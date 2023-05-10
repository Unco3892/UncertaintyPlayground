# design the multi-modal approach in `R`
# let's apply this for a single modality which is tabular data
# afterwards, I can read the images and the tabular data and cast them into the
# same object

# load all the necessary libraries
library(tidyverse)

# read the data
train_data <-
  read_csv(here::here("data/metadata/train_data.csv"))

train <- head(train_data, 100)

# we scale the inputs (training data)
X_train <- scale(train[, c(1:4)])
y_train <- train$price
dim(y_train) <- c(length(y_train), 1) # add extra dimension to vector

# we take the transpose of these metrices for simplicity (but it's not necessary)
X_train <- as.matrix(X_train, byrow = TRUE)
X_train <- t(X_train)
y_train <- as.matrix(y_train, byrow = TRUE)
y_train <- t(y_train)


# write a function that provides the

# To generate matrices with random parameters, we need to first obtain the size
# (number of neurons) of all the layers in our neural-net. We’ll write a function to do that.
# Let’s denote n_x, n_h, and n_y as the number of neurons in input layer, hidden layer, and output layer respectively.

getLayerSize <- function(X, y, hidden_neurons, train = TRUE) {
  n_x <- dim(X)[1]
  n_h <- hidden_neurons
  n_y <- dim(y)[1]

  size <- list(
    "n_x" = n_x,
    "n_h" = n_h,
    "n_y" = n_y
  )

  return(size)
}

layer_size <- getLayerSize(X_train, y_train, hidden_neurons = 20)
layer_size


# we initialize the paramaters
initializeParameters <- function(X, list_layer_size) {
  m <- dim(data.matrix(X))[2]

  n_x <- list_layer_size$n_x
  n_h <- list_layer_size$n_h
  n_y <- list_layer_size$n_y

  W1 <- matrix(rnorm(n_h * n_x), nrow = n_h, ncol = n_x, byrow = TRUE) * 0.01
  b1 <- matrix(rep(0, n_h), nrow = n_h)
  W2 <- matrix(rnorm(n_y * n_h), nrow = n_y, ncol = n_h, byrow = TRUE) * 0.01
  b2 <- matrix(rep(0, n_y), nrow = n_y)

  params <- list(
    "W1" = W1,
    "b1" = b1,
    "W2" = W2,
    "b2" = b2
  )

  return(params)
}

init_params <- initializeParameters(X_train, layer_size)
lapply(init_params, function(x) dim(x))

# we can use the tanh function at this point

# we can also do forward propagation
forwardPropagation <- function(X, params, list_layer_size) {
  m <- dim(X)[2]
  n_h <- list_layer_size$n_h
  n_y <- list_layer_size$n_y

  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2

  b1_new <- matrix(rep(b1, m), nrow = n_h)
  b2_new <- matrix(rep(b2, m), nrow = n_y)

  Z1 <- W1 %*% X + b1_new
  A1 <- Z1
  Z2 <- W2 %*% A1 + b2_new
  A2 <- Z2

  cache <- list(
    "Z1" = Z1,
    "A1" = A1,
    "Z2" = Z2,
    "A2" = A2
  )

  return(cache)
}
