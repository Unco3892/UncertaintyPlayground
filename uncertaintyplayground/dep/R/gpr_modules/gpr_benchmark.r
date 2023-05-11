#' Compute mean squared error (MSE) between true and predicted values
#'
#' @param y_true A numeric vector of true values
#' @param y_pred A numeric vector of predicted values
#' @return A numeric value of the mean squared error between true and predicted values
#' @export
# Comparing the Gaussian Process Regression model to two simple linear regression models
# compute mse and other metrics for comparison
mse <- function(y_true, y_pred) {
  sum((y_true - y_pred)^2) / length(y_true)
}

#' Compare Gaussian Process Regression (GPR) model to two simple linear regression models
#'
#' @param train A data frame of training data
#' @param test A data frame of test data
#' @param gpr_model A TensorFlow Gaussian Process Regression model
#' @param n_tfd_samples An integer specifying the number of samples to draw from the GPR model
#' @param weights_vector A numeric vector of weights for weighted linear regression
#' @return A list containing two ggplot objects, and three numeric values: the mean squared errors for the two linear regression models and the GPR model
#' @export
compare_models <- function(train, test, gpr_model, n_tfd_samples = 1000, weights_vector = NULL) {
  # assume that the outcome variable is the last column
  outcome_col <- colnames(train)[ncol(train)]

  # simple linear model with no interactions
  fit1_formula <- as.formula(paste(outcome_col, "~ ."))
  fit1 <- lm(fit1_formula, data = train, weights = weights_vector)

  # two-way interactions
  fit2_formula <- as.formula(paste(outcome_col, "~ (.) ^ 2"))
  fit2 <- lm(fit2_formula, data = train, weights = weights_vector)

  # Make predictions with the linear regression models
  linreg_preds1 <- fit1 %>% predict(test[, 1:(ncol(test) - 1)])
  linreg_preds2 <- fit2 %>% predict(test[, 1:(ncol(test) - 1)])

  # Add results to a dataframe
  compare <-
    data.frame(
      y_true = pull(test, outcome_col),
      linreg_preds1 = linreg_preds1,
      linreg_preds2 = linreg_preds2
    )

  # set seed to make sure the same samples are generated
  tensorflow::set_random_seed(42)

  # Make predictions with the Gaussian Process Regression model
  yhats <- gpr_model(tf$convert_to_tensor(as.matrix(test[, 1:(ncol(test) - 1)])))
  tensorflow::set_random_seed(42)
  yhat_samples <- yhats %>%
    tfd_sample(n_tfd_samples) %>%
    tf$squeeze() %>%
    tf$transpose()
  sample_means <- yhat_samples %>% apply(1, mean)

  # Add the VGP predictions to the results
  compare <- compare %>%
    cbind(vgp_preds = sample_means)

  # plot the VPG predictions
  plot1 <- function() {
    ggplot(compare, aes(x = y_true)) +
      geom_abline(slope = 1, intercept = 0) +
      geom_point(aes(y = vgp_preds, color = "VGP")) +
      geom_point(aes(y = linreg_preds1, color = "simple lm"), alpha = 0.4) +
      geom_point(aes(y = linreg_preds2, color = "lm w/ interactions"), alpha = 0.4) +
      scale_colour_manual("",
        values = c("VGP" = "black", "simple lm" = "cyan", "lm w/ interactions" = "violet")
      ) +
      coord_cartesian(xlim = c(min(compare$y_true), max(compare$y_true)), ylim = c(min(compare$y_true), max(compare$y_true))) +
      ylab("predictions") +
      theme(aspect.ratio = 1, legend.position = "top")
  }

  lm_metrics1 <- caret::postResample(compare$linreg_preds1, compare$y_true)
  lm_metrics2 <- caret::postResample(compare$linreg_preds2, compare$y_true)
  vgp_metrics <- caret::postResample(compare$vgp_preds, compare$y_true)

  # samples_df <-
  #   data.frame(cbind(compare$y_true, as.matrix(yhat_samples))) %>%
  #   gather(key = run, value = prediction, -X1) %>%
  #   rename(y_true = "X1")

  # plot2 <- function() {
  #   samples_df <-
  #     data.frame(cbind(compare$y_true, as.matrix(yhat_samples))) %>%
  #     gather(key = run, value = prediction, -X1) %>%
  #     rename(y_true = "X1")

  #   median_df <- samples_df %>%
  #     group_by(y_true, run) %>%
  #     summarize(median_prediction = median(prediction))

  #   ggplot(samples_df, aes(y_true, prediction)) +
  #     geom_point(aes(color = run),
  #                alpha = 0.2,
  #                size = 2) +
  #     geom_point(data = median_df, aes(y_true, median_prediction, color = run),
  #                alpha = 0.8,
  #                size = 3) +
  #     geom_abline(slope = 1, intercept = 0) +
  #     theme(legend.position = "none") +
  #     ylab("repeated predictions") +
  #     theme(aspect.ratio = 1)
  # }
  plot2 <- function() {
    samples_df <-
      data.frame(cbind(compare$y_true, as.matrix(yhat_samples))) %>%
      gather(key = run, value = prediction, -X1) %>%
      rename(y_true = "X1")

    # Randomly select 10 runs
    random_runs <- sample(unique(samples_df$run), 50)

    # Filter the samples_df to include only the selected runs
    samples_df_subset <- samples_df %>%
      filter(run %in% random_runs)

    ggplot(samples_df_subset, aes(y_true, prediction)) +
      geom_point(aes(color = run),
        alpha = 0.2,
        size = 2
      ) +
      geom_abline(slope = 1, intercept = 0) +
      theme(legend.position = "none") +
      ylab("repeated predictions") +
      theme(aspect.ratio = 1)
  }
  return(list(plot1 = plot1, plot2 = plot2, metrics1 = lm_metrics1, metrics2 = lm_metrics2, vgp_metrics = vgp_metrics))
}
