install.packages("tabnet")

mod <- tabnet(
  epochs = 3, batch_size = 16384, decision_width = 24, attention_width = 26,
  num_steps = 5, penalty = 0.000001, virtual_batch_size = 512, momentum = 0.6,
  feature_reusage = 1.5, learn_rate = 0.02
) %>%
  set_engine("torch", verbose = TRUE) %>%
  set_mode("classification")

rec <- recipe(class ~ ., train)

library(tabnet)
library(recipes)
set.seed(1)
rec <- recipe(Y ~ ., data = df_train)
fit <-
  tabnet_fit(rec,
    df_train,
    epochs = 500,
    config = tabnet_config(loss = "mse", verbose = 1)
  )
autoplot(fit)

postResample(pred = predict(fit, df_train), obs = y_train)


postResample(pred = predict(fit, df_valid), obs = y_valid)

df_train

suppressWarnings(autoplot(fit))

train


# measure this performance on both the training and test sets
# training set
lm_mod_train <- lm(Y ~ ., data = df_train)
postResample(pred = predict(lm_mod_train), obs = y_train)
postResample(pred = infer_v3(df_train), obs = y_train)


# test set
postResample(
  pred = predict(lm_mod_train, newdata = df_test),
  obs = y_test
)
postResample(pred = infer_v3(df_test, size = nrow(df_test)), obs = y_test)
