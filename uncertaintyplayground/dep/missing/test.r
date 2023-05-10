library(missForest)
library(tidyverse)

diamonds_numeric <- diamonds %>% 
  select_if(is.numeric) %>% 
  filter(row_number()<100)

## The data contains four continuous and one categorical variable.

## Artificially produce missing values using the 'prodNA' function:
set.seed(81)
diamons.mis <- prodNA(diamonds_numeric, noNA = 0.1)
diamons.mis <- as.data.frame(diamons.mis)

## Impute missing values providing the complete matrix for
## illustration. Use 'verbose' to see what happens between iterations:
iris.imp <- missForest(diamons.mis, xtrue = diamonds_numeric)

iris.imp$imp

## The imputation is finished after five iterations having a final
## true NRMSE of 0.143 and a PFC of 0.036. The estimated final NRMSE
## is 0.157 and the PFC is 0.025 (see Details for the reason taking
## iteration 4 instead of iteration 5 as final value).

## The final results can be accessed directly. The estimated error:
iris.imp$OOBerror

## The true imputation error (if available):
iris.imp$error

