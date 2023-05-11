# data import and setup
#-------------------------------------------------------------------------------
library(tidyverse)
library(caret)
library(keras)
library(tensorflow)
# library(broom)
# library(nnet)
# library(parallel)
# library(doParallel)

#-------------------------------------------------------------------------------
# turn off the warnings
options(warnings = -1)

# general parameters
set.seed(1)
N <- 1000
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
