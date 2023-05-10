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

#--------------------------------------------------------------------------------
# testing gaussian process regression alone
# x_train <- select(df_train, -Y) %>% as.matrix()
# y_train <- df_train$Y
# reticulate::source_python(here::here("scripts/r/gpr_modules/gpytorch_modular.py"))
# gpr_mod = SparseGPTrainer(x_train, y_train, num_epochs=5000L, batch_size=250L, lr=0.4)
# gpr_mod$train()
# assign the two variables on the same line (similar to `,` in python)
# purrr::walk2(c('y_pred', 'y_var') ,gpr_mod$predict_with_uncertainty(gpr_mod$X_val),assign,envir =parent.frame()) 
# caret::postResample(pred = y_pred, obs = gpr_mod$y_val$numpy())
# purrr::walk2(c('y_pred', 'y_var') ,gpr_mod$predict_with_uncertainty(x_test),assign,envir =parent.frame()) 
# nice work
# caret::postResample(pred = y_pred, obs = y_test)
# I think I can easily beat this score but let's see