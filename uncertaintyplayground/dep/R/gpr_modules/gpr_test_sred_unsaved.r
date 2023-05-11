# Scripts to test `gpr_models.R` and `gpr_diagnostics.R` functions
# source(here::here("scripts/r/gpr_modules/gpr_models.r"))
source(here::here("scripts/r/gpr_modules/while_lm_gpr.r"))

library(rsample)

# Applying this to the SRED data
# train-test split
set.seed(777)
dummy_data <- data.frame(hidden_node, y_comp, w_comp)

split <- initial_split(dummy_data, prop = 0.8)
train <- training(split)
test <- testing(split)

train_sample_weights <- pull(train, w_comp)
train <- select(train, -w_comp)
test <- select(test, -w_comp)

X_train <- train[,-ncol(train)] %>% as.matrix()
y_train <- train$y_comp
X_test <- test[,-ncol(test)] %>% as.matrix()
y_test <- test$y_comp

# SparseGPTrainer <- reticulate::import_from_path(module = "gpytorch_modular", path = here::here("scripts/r/gpr_modules/"))
# the higher the batch size, the better the results
# reticulate::source_python(here::here("scripts/r/gpr_modules/gpytorch_modular.py"))
# gpr_mod = SparseGPTrainer(X_train, y_train, num_epochs=5000L, batch_size=2000L, lr=0.2)
# gpr_mod$train()
start.time <- Sys.time()
reticulate::source_python(here::here("scripts/r/gpr_modules/gpytorch_modular.py"), convert = T)
gpr_mod = SparseGPTrainer(X_train, y_train, sample_weights = train_sample_weights, num_inducing_points = 200L, num_epochs=5000L, batch_size=1000L, lr=0.6, patience = 3L) # , min_delta = 0.00001 #, use_scheduler = T
gpr_mod$train()
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
# Better performance (`caret::postResample()`) if `convert=True` since we get less precision for the floating points

start.time <- Sys.time()
reticulate::source_python(here::here("scripts/r/gpr_modules/gpytorch_modular.py"), convert = T)
gpr_mod = SparseGPTrainer(X_train, y_train, sample_weights = train_sample_weights, num_inducing_points = 75L, num_epochs=5000L, batch_size=1000L, lr=0.1, patience = 2L) # , 
gpr_mod$train()
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

# best performance + speed
reticulate::source_python(here::here("scripts/r/gpr_modules/gpytorch_modular.py"), convert = T)
gpr_mod = SparseGPTrainer(X_train, y_train, sample_weights = train_sample_weights, num_inducing_points = 75L, num_epochs=5000L, batch_size=1000L, lr=0.1, patience = 2L) # , 
gpr_mod$train()

# fast (and accurate enough model)
# reticulate::source_python(here::here("scripts/r/gpr_modules/gpytorch_modular.py"), convert = T)
# gpr_mod = SparseGPTrainer(X_train, y_train, sample_weights = train_sample_weights, num_inducing_points = 50L, num_epochs=5000L, batch_size=1000L, lr=0.1, patience = 2L) # , 
# gpr_mod$train()

# best model with weights
# reticulate::source_python(here::here("scripts/r/gpr_modules/gpytorch_modular.py"), convert = T)
# gpr_mod = SparseGPTrainer(X_train, y_train, sample_weights = train_sample_weights, num_inducing_points = 200L, num_epochs=5000L, batch_size=1000L, lr=0.15, patience = 2L) # , 
# gpr_mod$train()


#best setting with weights
# reticulate::source_python(here::here("scripts/r/gpr_modules/gpytorch_modular.py"), convert = T)
# gpr_mod = SparseGPTrainer(X_train, y_train, sample_weights = train_sample_weights, num_inducing_points = 200L, num_epochs=5000L, batch_size=1000L, lr=0.15, patience = 3L) # , 
# gpr_mod$train()

#---------------------------------------------------------------------------------------------------
# to expriment with varying learning rate
# reticulate::source_python(here::here("scripts/r/gpr_modules/gpytorch_modular.py"), convert = T)
# gpr_mod = SparseGPTrainer(X_train, y_train, sample_weights = train_sample_weights, num_inducing_points = 200L, num_epochs=5000L, batch_size=1000L, lr=0.1, patience = 3L, use_scheduler = T) # , 
# gpr_mod$train()


#---------------------------------------------------------------------------
purrr::walk2(c('y_pred', 'y_var') ,gpr_mod$predict_with_uncertainty(gpr_mod$X_val),assign,envir =parent.frame())
caret::postResample(pred = y_pred, obs = py_to_r(gpr_mod$y_val$numpy()))


# purrr::walk2(c('y_pred', 'y_var') ,reticulate::py_to_r(gpr_mod$predict_with_uncertainty(gpr_mod$X_val)),assign,envir =parent.frame()) 
# caret::postResample(pred = y_pred, obs = py_to_r(gpr_mod$y_val$numpy()))

a <- gpr_mod$predict_with_uncertainty(gpr_mod$X_val$numpy())[[1]]

a %>% str()
caret::postResample(pred = as.vector(a), obs = as.vector(gpr_mod$y_val$numpy()))

# increasing the batch size drastically improves the performance but slows the training

# Improvements
# DONE:a) Implement the modular optimizer
# ALSO IMPLEMENT A LEARNING RATE SCHEDULERs
# b) Implemenet it in the while loop & compare performances
# c) Implement it into the prediction function
# d) Take care of the plotting
# f) Add some docstrings very later on
# DONE: g) Add `cuda` capabilities?
# Couldn't do it: Add the ability so that if the training only improves by extremely small margin, we STOP the training.
# Need to better understand the role of the batches in the current implementation

# ADD parallelization for the inference function -> Super fast, so not needed


#---------------------------------------------
# remotes::install_github("joshuaulrich/microbenchmark")
# install.packages(microbenchmark)

# benchmark code
# time_me <- function(){
#     reticulate::source_python(here::here("scripts/r/gpr_modules/gpytorch_modular.py"))
#     gpr_mod = SparseGPTrainer(X_train, y_train, num_inducing_points = 200L, num_epochs=5000L, batch_size=500L, lr=0.2, patience = 3L)
#     gpr_mod$train()
# }
# microbenchmark::microbenchmark(time_me(), times = 3)

# this could actually be nice
start.time <- Sys.time()
gpr_mod = SparseGPTrainer(X_train, y_train, optimizer_fn_name = "RMSprop", num_epochs=5000L, batch_size=2000L, lr=0.1)
gpr_mod$train()
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

# assign the two variables on the same line (similar to `,` in python)
# purrr::walk2(c('y_pred', 'y_var') ,gpr_mod$predict_with_uncertainty(gpr_mod$X_val),assign,envir =parent.frame()) 
# caret::postResample(pred = y_pred, obs = gpr_mod$y_val$numpy())
purrr::walk2(c('y_pred', 'y_var') ,gpr_mod$predict_with_uncertainty(X_test),assign,envir =parent.frame()) 
# nice work
caret::postResample(pred = y_pred, obs = y_test)


purrr::walk2(c('y_pred', 'y_var') ,gpr_mod$predict_with_uncertainty(gpr_mod$X_val),assign,envir =parent.frame()) 
caret::postResample(pred = y_pred, obs = gpr_mod$y_val$numpy())
# y_pred %>% as.vector() %>% str()
# y_var %>% as.vector() %>% str()

# nice work
# the code below gives the same output
# caret::postResample(pred = as.vector(y_pred), obs = as.vector(gpr_mod$y_val$numpy()))

# make the optimizer also optional so that you can give it RMSprop instead

gpr_mod$predict_with_uncertainty(X_test)

#---------------------------------------------
# applying the same thing but this time with sample weights
gpr_mod = SparseGPTrainer(X_train, y_train, sample_weights = train_sample_weights, num_epochs=5000L, batch_size=1000L, lr=0.2)
gpr_mod$train()
# assign the two variables on the same line (similar to `,` in python)
purrr::walk2(c('y_pred', 'y_var') ,gpr_mod$predict_with_uncertainty(X_test),assign,envir =parent.frame()) 
# nice work
caret::postResample(pred = y_pred, obs = y_test)

# purrr::walk2(c('y_pred', 'y_var') ,gpr_mod$predict_with_uncertainty(gpr_mod$X_val),assign,envir =parent.frame()) 
# caret::postResample(pred = y_pred, obs = gpr_mod$y_val$numpy())

# It doesn't work with X_test as it is
# also on the test set
purrr::walk2(c('y_pred', 'y_var') ,gpr_mod$predict_with_uncertainty(X_test),assign,envir =parent.frame()) 
# nice work
caret::postResample(pred = y_pred, obs = x_test)




#---------------------------------------------
# Adapt the plotting functions to this



#-------------------
# change all the code below to the pytorch approach
source(here::here("scripts/r/gpr_modules/gpr_models.r"))
# with the weights defined directly in the
tensorflow::set_random_seed(42)
# gpr_model_keras_weights <- gpr(data.frame(train[1:2000,]), data.frame(test), num_inducing_points = 200, num_epochs = 300, defined_lr = 0.1, n_tfd_samples = 100)

gpr_model_keras_weights <- gpr(train[1:2000,], test, num_inducing_points = 200, num_epochs = 300, defined_lr = 0.1)


# gpr_model_keras_weights <- gpr(train[1:10000, ], test, num_epochs = 300, defined_lr = 0.01, n_tfd_samples = 100, weights_vector = train_sample_weights[1:10000], keras_weighted_mode = T)

gpr_model_keras_weights <- gpr(df_train, df_test, num_inducing_points = 200, num_epochs = 300, defined_lr = 0.008, n_tfd_samples = 100)

gpr_model_keras_weights$history

test %>% str()
# with the weights defined directly in the loss function
# gpr_model_loss_weights <- gpr(train[1:1000, ], test, num_epochs = 300, defined_lr = 0.01, n_tfd_samples = 100, weights_vector = train_sample_weights[1:1000], keras_weighted_mode = F)
# gpr_model_loss_weights$history

#-----------------------------------------------------------------------------------
# to debug if there were any issues with the tensorflow dataset
# source(here::here("scripts/r/gpr_modules/gpr_utilities.r"))
# gpr_model_keras_weights <- gpr(train[1:1000, ], test, weights_vector = train_sample_weights[1:1000], num_epochs = 300, defined_lr = 0.01, n_tfd_samples = 100, keras_weighted_mode = T)
# features <- extract_component(gpr_model_keras_weights, "features")
# features %>% str()
# labels <- extract_component(gpr_model_keras_weights, "labels")
# labels %>% str()
# weights <- extract_component(gpr_model_keras_weights, "weights")
# weights %>% str()

#-----------------------------------------------------------------------------------
source(here::here("scripts/r/gpr_modules/gpr_diagnostics.r"))

tensorflow::set_random_seed(42)
# Apply the functions and plot the results
final_model <- gpr_model_keras_weights
from_df <- test
# Plot the density for any instance using the plot_predict_distribution function
ind <- 30
instance_pred <- predict_distribution(final_model$model, from_df, instance_index = ind)
plot_predict_distribution(instance_pred)

# Plot the confidence intervals using the gpr_confidence_intervals function
y_true <- pull(test, ncol(test))
preds <- gpr_confidence_intervals(y_true, final_model$distributions)
# Plot the true target values, predicted mean, and confidence intervals
plot_gpr_confidence_intervals(preds)

# FIX THIS PLOT
# Create the plot using the gpr_mean_errorbars function
plot_gpr_mean_errorbars(preds)

# Call the plot_gpr_densities function for any specific predictions
indices <- c(1, 80, 90, 91, 200)
plot_gpr_densities(final_model$distributions, indices)

#-----------------------------------------------------------------------------------
source(here::here("scripts/r/gpr_modules/gpr_benchmark.r"))
# Run and compare_models function using the train, test, and Gaussian Process Regression model
result <- compare_models(train, test, final_model$model, n_tfd_samples = 100)
result <- compare_models(train, test, final_model$model, n_tfd_samples = 100, weights_vector = train_sample_weights)

# train_sample_weights <- runif(nrow(train))*10
# much better results than the Rstudio post, why?
# compare_models(train, test, final_model$model)

# Display the plots
result$plot1()
result$plot2() # this is done for 100 samples

# Display the MSE values
cat(
    "Simple linear model -> ",
    paste0(names(round(result$metrics1, 3)), ": ", round(result$metrics1, 3), collapse = ", "), "\n"
)
cat(
    "Simple linear model -> ",
    paste0(names(round(result$metrics2, 3)), ": ", round(result$metrics2, 3), collapse = ", "), "\n"
)
cat(
    "Gaussian Process Regression -> ",
    paste0(names(round(result$vgp_metrics, 3)), ": ", round(result$vgp_metrics, 3), collapse = ", "), "\n"
)
#-----------------------------------------------------------------------------------
# Always take the largest value of diffences to calculate the min_delta
## You should consider the improvements in the validation MSE during the epochs following epoch 19:
## Epoch 20: 0.022851 - 0.022845 = 0.000006
## Epoch 21: 0.022851 - 0.022860 = -0.000009
## Epoch 22: 0.022851 - 0.022889 = -0.000038
## Epoch 23: 0.022851 - 0.022927 = -0.000076
## Epoch 24: 0.022851 - 0.022969 = -0.000118
## The largest improvement in this range is 0.000006 (between epochs 19 and 20). To make sure that the training stops after epoch 19, set the min_delta value to slightly greater than the largest improvement, for example:

