# data import and setup
#-------------------------------------------------------------------------------
library(tidyverse)
library(caret)
library(keras)
library(tensorflow)
library(abind)

# ------------------------------------------------------------------------------
# data splitting parameters
train_smp_size <- floor(train_size * N)
train_ind <- sample(seq_len(N), size = train_smp_size)
valid_smp_size <- floor(valid_size * N)
valid_ind <- sample(seq_len(train_smp_size), size = valid_smp_size)
train_smp_size <- train_smp_size - valid_smp_size

# ------------------------------------------------------------------------------
# working on the SRED (real) data
## loading the data
df <- read_csv(here::here("data/metadata/train_data.csv"))
df <- scale(df[, c(2:5)]) %>%
  bind_cols(Y = log(df$price)) %>%
  mutate(img_path = paste0(
    here::here("data/processed_images/train/montage_organized//"),
    as.character(df$listing_id),
    ".jpeg"
  ))
# df <-  df[complete.cases(df),]
df_smp_idx <- sample(nrow(df), size = N)
df <- df[df_smp_idx, ]

# ------------------------------------------------------------------------------
# load the helper functions including the one for the images
source("scripts/r/utilities/helpers.R")

# ------------------------------------------------------------------------------
# lazy load the training and test data especially for the images
# divide between train and test
df_train <- df[train_ind, ]
df_valid <- df_train[valid_ind, ]
df_train <- df_train[-valid_ind, ]
df_test <- df[-train_ind, ]

# delayed assign results in variables being loaded only once they're called
## process tabular data + load the images
# remove the id from all the tabular predictions
delayedAssign("x_train_img", {
  load_imgs_from_dir(df_train$img_path, target_size = img_h_w)
})

delayedAssign("x_train_tab", {
  select(df_train, -c(Y, img_path)) %>% as.matrix()
})

delayedAssign("y_train", {
  df_train$Y %>% as.matrix()
})

delayedAssign("x_valid_img", {
  load_imgs_from_dir(df_valid$img_path, target_size = img_h_w)
})

delayedAssign("x_valid_tab", {
  select(df_valid, -c(Y, img_path)) %>% as.matrix()
})

delayedAssign("y_valid", {
  df_valid$Y %>% as.matrix()
})

delayedAssign("x_test_img", {
  load_imgs_from_dir(df_test$img_path, target_size = img_h_w)
})

delayedAssign("x_test_tab", {
  select(df_test, -c(Y, img_path)) %>% as.matrix()
})

delayedAssign("y_test", {
  df_test$Y %>% as.matrix()
})

# delayed assign results in variables being loaded only once they're called

# test_prop_tensors <-
#   readRDS("data/final_test_224_image_tensors_organized.rds") %>%
#   keras_array()
# test_prop_tensors %>% str()
