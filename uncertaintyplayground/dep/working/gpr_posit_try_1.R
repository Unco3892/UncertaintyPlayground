# taken from https://blogs.rstudio.com/ai/posts/2019-12-10-variational-gaussian-process/
#tf$executing_eagerly()

library(tidyverse)
library(readxl)
library(rsample)
library(reticulate)
library(tfdatasets)
library(keras)
library(tfprobability)


# get the data
concrete <- read_xls(
  "scripts/working/Concrete_Data.xls",
  col_names = c(
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
    "strength"
  ),
  skip = 1
)

concrete %>% glimpse()
cement_ <- cut(concrete$cement, 3, labels = c("low", "medium", "high"))
fit <- lm(strength ~ (.) ^ 2, data = cbind(concrete[, 2:9], cement_))
summary(fit)

# scale predictors here already, so data are the same for all models
concrete[, 1:8] <- scale(concrete[, 1:8])

# train-test split 
set.seed(777)
split <- initial_split(concrete, prop = 0.8)
train <- training(split)
test <- testing(split)

# simple linear model with no interactions
fit1 <- lm(strength ~ ., data = train)
fit1 %>% summary()

# two-way interactions
fit2 <- lm(strength ~ (.) ^ 2, data = train)
fit2 %>% summary()

linreg_preds1 <- fit1 %>% predict(test[, 1:8])
linreg_preds2 <- fit2 %>% predict(test[, 1:8])

compare <-
  data.frame(
    y_true = test$strength,
    linreg_preds1 = linreg_preds1,
    linreg_preds2 = linreg_preds2
  )

create_dataset <- function(df, batch_size, shuffle = TRUE) {
  df <- as.matrix(df)
  ds <-
    tensor_slices_dataset(list(df[, 1:8], df[, 9, drop = FALSE]))
  if (shuffle)
    ds <- ds %>% dataset_shuffle(buffer_size = nrow(df))
  ds %>%
    dataset_batch(batch_size = batch_size)
}

# just one possible choice for batch size ...
batch_size <- 64
train_ds <- create_dataset(train, batch_size = batch_size)
test_ds <- create_dataset(test, batch_size = nrow(test), shuffle = FALSE)


# define the class for the variational layer
k_set_floatx("float64")
bt <- import("builtins")
RBFKernelFn <- reticulate::PyClass(
  "KernelFn",
  inherit = tensorflow::tf$keras$layers$Layer,
  list(
    `__init__` = function(self, dtype = NULL, trainable = TRUE, ...) {
      super()$`__init__`(trainable = trainable, ...)
      self$`_amplitude` = self$add_variable(initializer = initializer_zeros(),
                                            dtype = dtype,
                                            name = 'amplitude')
      self$`_length_scale` = self$add_variable(initializer = initializer_zeros(),
                                               dtype = dtype,
                                               name = 'length_scale')
      NULL
    },
    
    call = function(self, x, ...) {
      x
    },
    
    kernel = bt$property(
      reticulate::py_func(
        function(self)
          tfp$math$psd_kernels$ExponentiatedQuadratic(
            amplitude = tf$nn$softplus(array(0.1) * self$`_amplitude`),
            length_scale = tf$nn$softplus(array(2) * self$`_length_scale`)
          )
      )
    )
  )
)

# inducing points
num_inducing_points <- 50

sample_dist <- tfd_uniform(low = 1, high = nrow(train) + 1)
sample_ids <- sample_dist %>%
  tfd_sample(num_inducing_points) %>%
  tf$cast(tf$int32) %>%
  as.numeric()

sampled_points <- train[sample_ids, 1:8]

model <- keras_model_sequential() %>%
  layer_dense(units = 8,
              input_shape = 8,
              use_bias = FALSE) %>%
  layer_variational_gaussian_process(
    num_inducing_points = num_inducing_points,
    kernel_provider = RBFKernelFn(),
    event_shape = 1,
    inducing_index_points_initializer = initializer_constant(as.matrix(sampled_points)),
    unconstrained_observation_noise_variance_initializer =
      initializer_constant(array(0.1))
  )

# KL weight sums to one for one epoch
kl_weight <- batch_size / nrow(train)

# loss that implements the VGP algorithm
loss <- function(y, rv_y)
  rv_y$variational_loss(y, kl_weight = kl_weight)

model %>% compile(optimizer = optimizer_adam(learning_rate = 0.008),
                  loss = loss,
                  metrics = "mse")

history <- model %>% fit(train_ds,
                         epochs = 200,
                         validation_data = test_ds)

plot(history)

# Make predictions with the new model
yhats <- model(tf$convert_to_tensor(as.matrix(test[, 1:8])))
yhat_samples <-  yhats %>%
  tfd_sample(10) %>%
  tf$squeeze() %>%
  tf$transpose()
sample_means <- yhat_samples %>% apply(1, mean)
compare <- compare %>%
  cbind(vgp_preds = sample_means)

# plot the VPG predictions
ggplot(compare, aes(x = y_true)) +
  geom_abline(slope = 1, intercept = 0) +
  geom_point(aes(y = vgp_preds, color = "VGP")) +
  geom_point(aes(y = linreg_preds1, color = "simple lm"), alpha = 0.4) +
  geom_point(aes(y = linreg_preds2, color = "lm w/ interactions"), alpha = 0.4) +
  scale_colour_manual("", 
                      values = c("VGP" = "black", "simple lm" = "cyan", "lm w/ interactions" = "violet")) +
  coord_cartesian(xlim = c(min(compare$y_true), max(compare$y_true)), ylim = c(min(compare$y_true), max(compare$y_true))) +
  ylab("predictions") +
  theme(aspect.ratio = 1) 

# compute mse and other metrics for comparison
mse <- function(y_true, y_pred) {
  sum((y_true - y_pred) ^ 2) / length(y_true)
}

lm_mse1 <- mse(compare$y_true, compare$linreg_preds1) # 117.3111
lm_mse2 <- mse(compare$y_true, compare$linreg_preds2) # 80.79726
vgp_mse <- mse(compare$y_true, compare$vgp_preds)     # 58.49689

samples_df <-
  data.frame(cbind(compare$y_true, as.matrix(yhat_samples))) %>%
  gather(key = run, value = prediction, -X1) %>% 
  rename(y_true = "X1")

ggplot(samples_df, aes(y_true, prediction)) +
  geom_point(aes(color = run),
             alpha = 0.2,
             size = 2) +
  geom_abline(slope = 1, intercept = 0) +
  theme(legend.position = "none") +
  ylab("repeated predictions") +
  theme(aspect.ratio = 1)