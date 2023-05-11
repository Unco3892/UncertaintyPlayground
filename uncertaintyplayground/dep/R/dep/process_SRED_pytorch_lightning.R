######################################################################
################ preparing SRED for google colab expriments############
######################################################################
library(tidyverse)
library(magrittr)
library(keras)
library(raster)
library(tensorflow)
library(foreach)
library(doSNOW)
library(doParallel)

train_data <-
  read_csv(here::here("data/metadata/train_data.csv"))

test_data <-
  read_csv(here::here("data/metadata/test_data.csv"))

source_root <- here::here("data/images/listings//")
destination_root <- here::here("data/processed_images//")


# file.copy(
# create directories for each listing
move_files <-
  function(listing_id,
           data_type,
           category_to_copy) {
    file.rename(
      paste0(
        source_root,
        data_type,
        "/",
        listing_id,
        "/",
        category_to_copy,
        ".jpeg"
      ),
      paste0(
        destination_root,
        data_type,
        "/",
        category_to_copy,
        "/",
        listing_id,
        ".jpeg"
      )
    )
  }

# category_to_copy <- "montage_organized"
# data_type <- "train"
# target_files <- pull(train_data, listing_id)
# for (j in seq_along(target_files)) {
#   move_files(target_files[j], data_type, category_to_copy)
# }

data_types <- list("train", "test")
categories_to_copy <- list("montage_random", "satellite", "cat")

# # three level loop to move everything
# for (d in seq_along(data_types)) {
#   target_files <-
#     pull(`if`(data_types[d] == "train", train_data, test_data),
#          listing_id)
#   for (h in seq_along(categories_to_copy)) {
#     for (j in seq_along(target_files)) {
#       move_files(target_files[j], data_types[d], categories_to_copy[h])
#     }
#   }
# }
