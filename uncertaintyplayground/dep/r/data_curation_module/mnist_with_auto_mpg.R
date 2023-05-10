library(tidyverse)
library(keras)
library(OpenML)

# get the AutoMPG data
# check if you don't have it locally, then download it again
url_autompg <- "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
filename_auto_mpg <- here::here("data", basename(url_autompg))
if (!file.exists(filename_auto_mpg)) {
  download.file(url_autompg, filename_auto_mpg, method = "curl")
}
auto_mpg <- read.table(filename_auto_mpg, header = FALSE, dec = ",")
auto_mpg_names <- c("mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name")
colnames(auto_mpg) <- auto_mpg_names

# add a tru catch if it fails
url_mnist <- "https://datahub.io/machine-learning/mnist_784/r/mnist_784.csv"
filename_mnist <- here::here("data", basename(url_mnist))
if (!file.exists(filename_mnist)) {
  download.file(url_mnist, filename_auto_mpg, method = "curl")
}

# make the cylinders into factor
auto_mpg$cylinders <- as.factor(auto_mpg$cylinders)

# get the mnist data
# mnist_data <- getOMLDataSet(data.id = 554L)
# colnames(mnist_data$data) <- mnist_data$colnames.new
# mnist_data <- mnist_data$data

mnist_data <- read.table(url_mnist, header = T, sep = ",") %>% tibble()
mnist_data
mnist_data$class <- as.factor(mnist_data$class)

# assign a row number by the cylinder id
auto_mpg %<>%
  group_by(cylinders) %>%
  mutate(row_id = row_number()) %>%
  ungroup()

# Randomly shuffle the mnist digits
# mnist_data <- mnist_data[sample(nrow(mnist_data)), ]
# mnist_data %>%
#   group_by(class) %>%
#   filter(row_id <= max_row_id$max_row_id[class])

# obtain the maximum row_id value for each group in df1
max_row_id <- auto_mpg %>%
  group_by(cylinders) %>%
  summarize(max_row_id = max(row_id))

set.seed(1)
# keep only the digits that are in the other dataframe
mnist_data %<>%
  filter(class %in% levels(auto_mpg$cylinders)) %>%
  sample_n(size = nrow(.)) %>%
  group_by(class) %>%
  mutate(row_id = row_number()) %>%
  ungroup() %>%
  mutate(class = droplevels(class))

# check this code to see if it did the right thing or not
mnist_data <- map_dfr(unique(mnist_data$class), function(c) {
  max_row_id_c <- max_row_id$max_row_id[max_row_id$cylinders == c]
  filter(mnist_data, class == c, row_id <= max_row_id_c)
})

# joining the dataframe
auto_mpg %<>%
  left_join(mnist_data, by = c("cylinders" = "class", "row_id" = "row_id"))

# even a faster way to change for unique instances
auto_mpg %>%
  unite(pixel_values, starts_with("pixel"), sep = ",") %>% # concatenate pixel columns into a single column
  filter(duplicated(pixel_values)) # filter out rows with duplicate pixel values

# Check if all pixel columns are unique for each instance
# auto_mpg %>%
#   rowwise() %>%
#   mutate(pixel_cols = paste0(c_across(starts_with("pixel")), collapse = ",")) %>%
#   select(pixel_cols) %>%
#   distinct() %>%
#   nrow() == nrow(auto_mpg)

plot_mnist_image <- function(df_input, instance_n, n_pixels = 28) {
  df_input %<>% dplyr::slice(instance_n)
  df_input_pixels <- df_input %>% select(starts_with("pixel"))
  image_matrix <- matrix(df_input_pixels, nrow = 28, byrow = TRUE)
  im_numbers <- apply(image_matrix, 2, as.numeric)
  im_numbers <- t(apply(im_numbers, 2, rev))
  image(1:n_pixels, 1:n_pixels, im_numbers, col = gray((0:255) / 255))
  return(select(df_input, -starts_with("pixel")))
}

plot_mnist_image(auto_mpg, 380)
