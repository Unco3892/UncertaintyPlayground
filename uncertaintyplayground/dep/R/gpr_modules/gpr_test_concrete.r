# Scripts to test `gpr_models.R` and `gpr_diagnostics.R` functions

source(here::here("scripts/r/gpr_modules/gpr_models.r"))

## data from Concrete Compressive Strength Data Set
## https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength
## get some data and apply the function
concrete <- read_xls(
    "data/Concrete_Data.xls",
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

cement_ <- cut(concrete$cement, 3, labels = c("low", "medium", "high"))
# scale predictors here already, so data are the same for all models
concrete[, 1:8] <- scale(concrete[, 1:8])
# train-test split
set.seed(777)
split <- initial_split(concrete, prop = 0.8)
train <- training(split)
test <- testing(split)

# define some arbitary weights for testing
train_sample_weights <- runif(nrow(train))*10

X <- train[,-ncol(train)] %>% as.matrix()
y <- train$strength

# train[,-ncol(train)] %>% str()

#tensorflow::set_random_seed(42)
gpr_result_1 <- gpr(train, test, batch_size = 64, num_inducing_points = 50, early_stopping_patience = 20)
gpr_result_1$history



# tensorflow::set_random_seed(42)
# gpr_result_2 <- gpr(train, test, batch_size = 64, num_inducing_points = 50, early_stopping_patience = 20)
# identical(gpr_result_1$history, gpr_result_2$history)

# tensorflow::set_random_seed(42)
# gpr_result_3 <- gpr(train, test, weights_vector = train_sample_weights, keras_weighted_mode = F, batch_size = 64, num_inducing_points = 50, early_stopping_patience = 20)
# tensorflow::set_random_seed(42)
# gpr_result_4 <- gpr(train, test, weights_vector = train_sample_weights, keras_weighted_mode = F, batch_size = 64, num_inducing_points = 50, early_stopping_patience = 20)
# identical(gpr_result_3$history, gpr_result_4$history)

# tensorflow::set_random_seed(42)
# gpr_result_5 <- gpr(train, test, weights_vector = train_sample_weights, keras_weighted_mode = T, batch_size = 64, num_inducing_points = 50, early_stopping_patience = 20)
# tensorflow::set_random_seed(42)
# gpr_result_6 <- gpr(train, test, weights_vector = train_sample_weights, keras_weighted_mode = T, batch_size = 64, num_inducing_points = 50, early_stopping_patience = 20)
# identical(gpr_result_5$history, gpr_result_6$history)

# gpr_result_1$history
# gpr_result_3$history
# gpr_result_5$history

#-----------------------------------------------------------------------------------

source(here::here("scripts/r/gpr_modules/gpr_diagnostics.r"))

tensorflow::set_random_seed(42)
# Apply the functions and plot the results
final_model <- gpr_result_1
from_df <- test
# Plot the density for any instance using the plot_predict_distribution function
ind = 30
instance_pred <- predict_distribution(final_model$model,from_df, instance_index = ind)
plot_predict_distribution(instance_pred)

# Plot the confidence intervals using the gpr_confidence_intervals function
y_true <- pull(test,ncol(test))
preds <- gpr_confidence_intervals(y_true, final_model$distributions)
# Plot the true target values, predicted mean, and confidence intervals
plot_gpr_confidence_intervals(preds)

# Create the plot using the gpr_mean_errorbars function
plot_gpr_mean_errorbars(preds)

# Call the plot_gpr_densities function for any specific predictions
indices <- c(1, 80, 90, 91, 200)
plot_gpr_densities(final_model$distributions, indices)

#-----------------------------------------------------------------------------------
source(here::here("scripts/r/gpr_modules/gpr_benchmark.r"))
# Run and compare_models function using the train, test, and Gaussian Process Regression model
result <- compare_models(train, test, final_model$model)
# train_sample_weights <- runif(nrow(train))*10
# much better results than the Rstudio post, why?
# compare_models(train, test, final_model$model)

# Display the plots
result$plot1()
result$plot2() #this is done for 100 samples

# Display the MSE values
cat("Simple linear model -> ", 
    paste0(names(round(result$metrics1, 3)), ": ", round(result$metrics1, 3), collapse = ", "), "\n")
cat("Simple linear model -> ", 
    paste0(names(round(result$metrics2, 3)), ": ", round(result$metrics2, 3), collapse = ", "), "\n")
cat("Gaussian Process Regression -> ", 
    paste0(names(round(result$vgp_metrics, 3)), ": ", round(result$vgp_metrics, 3), collapse = ", "), "\n")


#-----------------------------------------------------------------------------------